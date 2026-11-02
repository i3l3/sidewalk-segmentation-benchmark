"""Synthetic integration checks. Never counts as human-reviewed real data."""
from pathlib import Path
import tempfile
import numpy as np
from PIL import Image
from .schema import *
from .dataset import import_images,rows,write_rows,split_dataset,refresh_reviews,validate_frozen
from .labeling import import_mask

def make_synthetic(root):
    root=init(root);source=root/'synthetic_source';source.mkdir(exist_ok=True)
    rng=np.random.default_rng(42)
    for i in range(9):
        array=rng.integers(0,256,(64,64,3),dtype=np.uint8)
        Image.fromarray(array).save(source/f'{i}.png')
    data=import_images(root,source,'synthetic integration test','unknown','synthetic')
    for i,r in enumerate(data):r.update(group_id='synthetic_'+str(i),camera_view='pedestrian')
    write_rows(root,data);split_dataset(root,targets=(5,2,2),confirm_groups=True)
    for i,r in enumerate(rows(root)):
        mask=np.zeros((64,64),np.uint8);mask[:,12:52]=2;mask[:,8:12]=3;mask[:8]=1;mask[:2,:2]=255
        p=root/f'synthetic_mask_{i}.png';Image.fromarray(mask).save(p)
        import_mask(root,r['image_id'],p,'SYNTHETIC_TEST_FIXTURE',True,label_source='synthetic_test_only')
    return root

def smoke(root):
    import torch
    from .training import train,safe_loss
    from .evaluation import confusion,metrics
    from .direction import extract_features,rule_direction
    root=Path(root)
    if 'synthetic' not in str(root).lower():raise ValueError('Use a separate path containing synthetic')
    if (root/'data/dataset_metadata.csv').exists():raise ValueError('Use a new synthetic root per run')
    make_synthetic(root)
    settings={'size':32,'epochs':2,'microbatch':1,'effective_batch':2,'seed':42}
    history1=train(root,'A',settings,tiny=True,stop_after_epoch=1)
    assert len(history1)==1
    history2=train(root,'A',settings,tiny=True,resume=True)
    assert len(history2)==2 and [r['epoch'] for r in history2]==[1,2]
    checkpoint=torch.load(root/'checkpoints/A/latest.pt',map_location='cpu',weights_only=False)
    assert {'optimizer','scheduler','scaler','rng','epoch'}<=checkpoint.keys()
    from .training import predict_student
    summary=predict_student(root,'A')
    assert summary and summary[0]['n']>0
    logits=torch.randn(1,9,8,8,requires_grad=True);labels=torch.full((1,8,8),255)
    loss=safe_loss(logits,labels);assert float(loss.detach())==0.;loss.backward();assert torch.isfinite(logits.grad).all()
    # Unknown prediction cannot erase a difficult pixel from metric denominators.
    m=metrics(confusion(np.array([[2,2]],np.uint8),np.array([[2,255]],np.uint8)))
    assert m['per_class'][2]['iou']==.5
    mask=np.zeros((64,64),np.uint8);mask[4:30,20:45]=2
    f,connected=extract_features(mask);assert not connected.any() and rule_direction(f)=='No clear path'
    # Validate complete-epoch resumed training against an uninterrupted run.
    uninterrupted=dict(settings)
    full=train(root,'C',uninterrupted,tiny=True)
    fullstate=torch.load(root/'checkpoints/C/latest.pt',map_location='cpu',weights_only=False)['model']
    resumed=checkpoint['model']
    maxdiff=max(float((resumed[k]-fullstate[k]).abs().max()) for k in resumed)
    if maxdiff>1e-5:raise AssertionError(f'Resume drift: {maxdiff}')
    result={'status':'passed','synthetic_only':True,'real_training':False,'completed_epochs':2,
            'resume_max_weight_difference':maxdiff,'evaluation_generated':True,'all_ignore_loss_finite':True,
            'disconnected_direction_abstains':True,'device':'cuda' if torch.cuda.is_available() else 'cpu'}
    save_json(root/'smoke_result.json',result);print(result,flush=True);return result
