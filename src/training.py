from pathlib import Path
import contextlib
import hashlib
import json
import math
import random
import time
import numpy as np
from PIL import Image,ImageEnhance
from .schema import *
from .dataset import validate_frozen
from .evaluation import confusion,metrics

EXPERIMENTS={'A':('b0',False),'B':('b0',True),'C':('b2',False),'D':('b2',True)}
DEFAULTS={'seed':42,'size':512,'epochs':20,'lr':6e-5,'weight_decay':.01,'effective_batch':8,
          'patience':5,'microbatch':None,'warmup_fraction':.1}

def seed_all(seed):
    import torch
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)

def rng_state():
    import torch
    return {'python':random.getstate(),'numpy':np.random.get_state(),'torch':torch.get_rng_state(),
            'cuda':torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}

def restore_rng(state):
    import torch
    random.setstate(state['python']);np.random.set_state(state['numpy']);torch.set_rng_state(state['torch'])
    if torch.cuda.is_available() and state['cuda']:torch.cuda.set_rng_state_all(state['cuda'])

class SegmentationDataset:
    def __init__(self,root,records,size=512,augment=False):self.root=Path(root);self.records=records;self.size=size;self.augment=augment
    def __len__(self):return len(self.records)
    def __getitem__(self,index):
        import torch
        r=self.records[index];image=Image.open(self.root/r['image_path']).convert('RGB')
        mask=Image.fromarray(read_mask(self.root/r['mask_path'],(image.height,image.width)))
        if self.augment:
            scale=random.uniform(.75,1.0);w=max(1,int(image.width*scale));h=max(1,int(image.height*scale))
            x=random.randint(0,image.width-w);y=random.randint(0,image.height-h)
            image=image.crop((x,y,x+w,y+h));mask=mask.crop((x,y,x+w,y+h))
            if random.random()<.5:image=image.transpose(Image.Transpose.FLIP_LEFT_RIGHT);mask=mask.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            image=ImageEnhance.Brightness(image).enhance(random.uniform(.9,1.1))
        image=image.resize((self.size,self.size),Image.Resampling.BILINEAR)
        mask=mask.resize((self.size,self.size),Image.Resampling.NEAREST)
        x=np.asarray(image,dtype=np.float32)/255
        x=(x-np.array([.485,.456,.406],np.float32))/np.array([.229,.224,.225],np.float32)
        return {'pixel_values':torch.from_numpy(x.transpose(2,0,1).copy()),
                'labels':torch.from_numpy(np.asarray(mask,dtype=np.int64).copy())}

def safe_loss(logits,labels):
    import torch
    import torch.nn.functional as F
    logits=F.interpolate(logits.float(),size=labels.shape[-2:],mode='bilinear',align_corners=False)
    if not (labels!=255).any():return logits.sum()*0
    return F.cross_entropy(logits,labels,ignore_index=255)

def load_student(variant,root,tiny=False):
    from transformers import SegformerConfig,SegformerForSemanticSegmentation
    if tiny:
        config=SegformerConfig(num_labels=9,depths=[1]*4,hidden_sizes=[8,16,32,64],
            num_attention_heads=[1,2,4,8],decoder_hidden_size=16,sr_ratios=[8,4,2,1],
            id2label=dict(enumerate(CLASSES)),label2id={v:k for k,v in enumerate(CLASSES)},semantic_loss_ignore_index=255)
        return SegformerForSemanticSegmentation(config),{'tiny_synthetic_only':True}
    from huggingface_hub import HfApi
    mid='nvidia/mit-'+variant
    revisions_path=Path(root)/'checkpoints/pretrained_revisions.json'
    revisions=load_json(revisions_path) if revisions_path.exists() else {}
    revision=revisions.get(mid) or HfApi().model_info(mid).sha
    revisions[mid]=revision;save_json(revisions_path,revisions)
    model,loading=SegformerForSemanticSegmentation.from_pretrained(mid,revision=revision,num_labels=9,
        id2label=dict(enumerate(CLASSES)),label2id={v:k for k,v in enumerate(CLASSES)},
        semantic_loss_ignore_index=255,ignore_mismatched_sizes=True,output_loading_info=True,
        use_safetensors=False)
    if any(k.startswith('segformer.') for k in loading['missing_keys']):raise RuntimeError('Pretrained encoder weights missing')
    return model,{'model_id':mid,'revision':revision,'loading_info':loading,
                  'note':'ImageNet encoder with newly initialized segmentation head; not historical trained B0'}

def training_records(root,experiment):
    data=validate_frozen(root);variant,pseudo=EXPERIMENTS[experiment]
    train=[r for r in data if r['split']=='train' and r['label_verified']=='true' and r['status']=='reviewed']
    if not train:raise ValueError('Reviewed training labels are required even for pseudo-label experiments')
    if pseudo:
        selection=load_json(Path(root)/'metrics/teacher_selection.json')
        lookup={r['image_id']:r for r in data}
        for sample in selection['reviewed_selection_samples']:
            r=lookup.get(sample['image_id'])
            if not r or r['split'] not in ['train','validation'] or r['label_verified']!='true' or r['mask_sha']!=sample['mask_sha']:
                raise ValueError('Teacher selection labels changed; rerun Teacher selection')
        train += [r for r in data if r['split']=='train' and r['status']=='pseudo' and r['label_source']==selection['selected']
                  and r['mask_path'] and sha(Path(root)/r['mask_path'])==r['mask_sha']]
        if not any(r['status']=='pseudo' for r in train):raise ValueError('B/D require additional, distinct pseudo-labeled images')
    val=[r for r in data if r['split']=='validation']
    if not val or any(r['label_verified']!='true' for r in val):raise ValueError('All validation labels must be human reviewed')
    for r in train+val:
        a=read_mask(Path(root)/r['mask_path'],(int(r['image_height']),int(r['image_width'])))
        if not (a!=255).any():raise ValueError('All-ignore sample cannot enter formal training')
    return variant,train,val

def train(root,experiment,overrides=None,resume=True,tiny=False,stop_after_epoch=None):
    import torch
    from torch.utils.data import DataLoader
    root=init(root);config={**DEFAULTS,**(overrides or {})};seed_all(config['seed'])
    variant,training,validation=training_records(root,experiment)
    if not tiny and not torch.cuda.is_available():raise RuntimeError('GPU is required for formal training')
    device='cuda' if torch.cuda.is_available() else 'cpu'
    if config['microbatch'] is None:
        vram=torch.cuda.get_device_properties(0).total_memory/2**30 if device=='cuda' else 0
        config['microbatch']=2 if vram>=12 else 1
    micro=int(config['microbatch']);effective=int(config['effective_batch'])
    if effective%micro:raise ValueError('effective_batch must be divisible by microbatch')
    folder=root/'checkpoints'/experiment;folder.mkdir(parents=True,exist_ok=True)
    signature=hashlib.sha256(json.dumps({'config':config,'experiment':experiment,'tiny':tiny,
        'samples':[(r['image_id'],r['image_sha'],r['mask_sha']) for r in training+validation],
        'split_lock':sha(root/'data/splits/locked.json'),'class_schema':sha(root/'data/class_schema.json')},sort_keys=True).encode()).hexdigest()
    latest=folder/'latest.pt';manifest=folder/'run.json'
    if latest.exists() and not resume:raise ValueError('Checkpoint exists; use a new root/version or resume=True')
    if manifest.exists() and load_json(manifest)['signature']!=signature:raise ValueError('Resume configuration/data mismatch')
    model,origin=load_student(variant,root,tiny);model.to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=config['lr'],weight_decay=config['weight_decay'])
    steps_per_epoch=math.ceil(len(training)/effective);total=max(1,steps_per_epoch*config['epochs'])
    warmup=max(1,int(total*config['warmup_fraction']))
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda step:min((step+1)/warmup,max(0.,(total-step)/max(1,total-warmup))))
    scaler=torch.amp.GradScaler('cuda',enabled=device=='cuda')
    start_epoch=0;best=-1.;stale=0;history=[]
    if latest.exists():
        # This file was produced by this run; external arbitrary .pt files are not accepted.
        checkpoint=torch.load(latest,map_location='cpu',weights_only=False)
        if checkpoint['signature']!=signature:raise ValueError('Unrecognized resume checkpoint')
        model.load_state_dict(checkpoint['model']);optimizer.load_state_dict(checkpoint['optimizer'])
        for state in optimizer.state.values():
            for k,v in state.items():
                if torch.is_tensor(v):state[k]=v.to(device)
        scheduler.load_state_dict(checkpoint['scheduler']);scaler.load_state_dict(checkpoint['scaler'])
        start_epoch=checkpoint['epoch']+1;best=checkpoint['best'];stale=checkpoint['stale'];history=checkpoint['history']
        restore_rng(checkpoint['rng'])
    import platform,importlib.metadata
    environment={'python':platform.python_version(),'torch':torch.__version__,
                 'transformers':importlib.metadata.version('transformers'),'numpy':np.__version__,
                 'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(0) if device=='cuda' else None,'device':device}
    previous_run=load_json(manifest) if manifest.exists() else {}
    save_json(manifest,{'signature':signature,'config':config,'origin':origin,'experiment':experiment,
        'tiny_synthetic_only':tiny,'device':device,'train_images':len(training),'validation_images':len(validation),
        'precision':'float16 AMP' if device=='cuda' else 'float32','created_at':previous_run.get('created_at',now()),
        'environments':previous_run.get('environments',[])+[environment]})
    trainset=SegmentationDataset(root,training,config['size'],True);valset=SegmentationDataset(root,validation,config['size'])
    if stale>=config['patience']:return history
    for epoch in range(start_epoch,config['epochs']):
        begin=time.perf_counter();model.train();optimizer.zero_grad(set_to_none=True);losses=[]
        generator=torch.Generator().manual_seed(config['seed']+epoch)
        loader=DataLoader(trainset,batch_size=micro,shuffle=True,generator=generator,num_workers=0)
        accumulation=effective//micro
        for i,batch in enumerate(loader):
            labels=batch['labels'].to(device);pixels=batch['pixel_values'].to(device)
            ctx=torch.autocast('cuda',dtype=torch.float16) if device=='cuda' else contextlib.nullcontext()
            with ctx:loss=safe_loss(model(pixel_values=pixels).logits,labels)
            # Normalize the final partial accumulation window by its actual microbatch count.
            window=min(accumulation,len(loader)-(i//accumulation)*accumulation)
            scaler.scale(loss/window).backward();losses.append(float(loss.detach()))
            if (i+1)%accumulation==0 or i+1==len(loader):
                scaler.unscale_(optimizer);torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
                scaler.step(optimizer);scaler.update();scheduler.step();optimizer.zero_grad(set_to_none=True)
        model.eval();cm=np.zeros((9,10),np.int64);vl=[]
        with torch.no_grad():
            for batch in DataLoader(valset,batch_size=1,shuffle=False):
                labels=batch['labels'].to(device);out=model(pixel_values=batch['pixel_values'].to(device)).logits
                vl.append(float(safe_loss(out,labels)))
                pred=torch.nn.functional.interpolate(out.float(),labels.shape[-2:],mode='bilinear',align_corners=False).argmax(1)
                cm+=confusion(labels[0].cpu().numpy(),pred[0].cpu().numpy())
        score=metrics(cm)['miou']
        if score is None:raise ValueError('No valid validation pixels')
        improved=score>best
        if improved:best=score;stale=0
        else:stale+=1
        item={'epoch':epoch+1,'train_loss':float(np.mean(losses)),'val_loss':float(np.mean(vl)),
              'val_miou':score,'epoch_seconds':time.perf_counter()-begin,
              'peak_gpu_mb':torch.cuda.max_memory_allocated()/2**20 if device=='cuda' else None}
        history.append(item);save_json(folder/'history.json',history)
        if improved:model.save_pretrained(folder/'best',safe_serialization=True)
        snapshot={'signature':signature,'model':model.state_dict(),'optimizer':optimizer.state_dict(),
            'scheduler':scheduler.state_dict(),'scaler':scaler.state_dict(),'rng':rng_state(),
            'epoch':epoch,'best':best,'stale':stale,'history':history}
        temp=folder/'latest.tmp';torch.save(snapshot,temp);temp.replace(latest)
        print(experiment,item,flush=True)
        if stale>=config['patience'] or (stop_after_epoch and epoch+1>=stop_after_epoch):break
    return history

def predict_student(root,experiment,split='test',evaluate=True):
    import torch
    from transformers import SegformerForSemanticSegmentation
    from .evaluation import evaluate_predictions
    root=Path(root);data=validate_frozen(root);run=load_json(root/'checkpoints'/experiment/'run.json')
    if run['tiny_synthetic_only'] and 'synthetic' not in str(root).lower():raise ValueError('Synthetic checkpoint not permitted for real-data evaluation')
    device='cuda' if torch.cuda.is_available() else 'cpu'
    model=SegformerForSemanticSegmentation.from_pretrained(root/'checkpoints'/experiment/'best').to(device).eval()
    selected=[r for r in data if r['split']==split]
    if not selected or (evaluate and any(r['label_verified']!='true' for r in selected)):raise ValueError('Reviewed evaluation split required')
    folder=root/'outputs/masks'/experiment;folder.mkdir(parents=True,exist_ok=True);stats={}
    ds=SegmentationDataset(root,selected,run['config']['size'])
    with torch.no_grad():
        for i,r in enumerate(selected):
            x=ds[i]['pixel_values'][None].to(device)
            if device=='cuda':torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
            begin=time.perf_counter();out=model(pixel_values=x).logits.float()
            pred=torch.nn.functional.interpolate(out,(int(r['image_height']),int(r['image_width'])),mode='bilinear',align_corners=False).argmax(1)[0]
            if device=='cuda':torch.cuda.synchronize()
            stats[r['image_id']]={'seconds':time.perf_counter()-begin,'peak_gpu_mb':torch.cuda.max_memory_allocated()/2**20 if device=='cuda' else None}
            Image.fromarray(pred.cpu().numpy().astype(np.uint8)).save(folder/f'{r["image_id"]}.png')
    save_json(folder/'run_info.json',stats)
    if evaluate:return evaluate_predictions(root,experiment,folder,range(9),split,stats)
    return str(folder)
