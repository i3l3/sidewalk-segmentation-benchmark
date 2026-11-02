from pathlib import Path
import csv
import json
import shutil
import zipfile
import numpy as np
from PIL import Image
from .schema import *
from .dataset import rows,write_rows,validate_frozen,refresh_reviews

TEACHERS=['mask2former_swin_l','oneformer_dinat_l']

def import_mask(root,image_id,path,reviewer='',verified=False,color_map=None,label_source='manual'):
    root=Path(root);data=rows(root)
    matches=[r for r in data if r['image_id']==image_id]
    if len(matches)!=1:raise ValueError(f'Unknown or duplicate image_id: {image_id}')
    row=matches[0]
    if verified and not reviewer.strip():raise ValueError('Explicit reviewer required to mark reviewed')
    array=np.asarray(Image.open(path))
    if array.ndim==3:
        if color_map is None:raise ValueError('RGB labels require explicit RGB->ID mapping')
        mapped=np.full(array.shape[:2],255,np.uint8);matched=np.zeros(array.shape[:2],bool)
        for color,cid in color_map.items():
            rgb=tuple(int(x) for x in color.split(','));sel=(array[:,:,:3]==rgb).all(2)
            mapped[sel]=int(cid);matched|=sel
        if not matched.all():raise ValueError('Unmapped RGB label colors')
        array=mapped
    shape=(int(row['image_height']),int(row['image_width']))
    if array.ndim!=2 or array.shape!=shape or set(np.unique(array))-(set(range(9))|{255}):
        raise ValueError('Mask must have original shape and IDs 0..8 or 255')
    if not (array!=255).any():raise ValueError('All-ignore mask cannot be approved or used for training')
    folder='verified_labels' if verified else 'pseudo_labels/imported'
    dest=root/'data'/folder/f'{image_id}.png';dest.parent.mkdir(parents=True,exist_ok=True)
    Image.fromarray(array.astype(np.uint8)).save(dest)
    values,counts=np.unique(array[array!=255],return_counts=True)
    row.update(mask_path=str(dest.relative_to(root)),mask_sha=sha(dest),label_source=label_source,
               label_verified=str(verified).lower(),status='reviewed' if verified else 'pseudo',
               reviewer=reviewer if verified else '',reviewed_at=now() if verified else '',
               dominant_class=CLASSES[int(values[counts.argmax()])])
    write_rows(root,data);return row

def reject(root,image_id,reviewer):
    if not reviewer.strip():raise ValueError('Reviewer required')
    data=rows(root);r=next(r for r in data if r['image_id']==image_id)
    r.update(status='rejected',label_verified='false',reviewer=reviewer,reviewed_at=now());write_rows(root,data)

def generate_teacher_labels(root,legacy_bundle,keys=TEACHERS):
    """Call in a fresh process; preserve full native classes, including obstacle semantics."""
    import sys
    legacy_bundle=Path(legacy_bundle).resolve();sys.path.insert(0,str(legacy_bundle))
    import benchmark_core as legacy
    root=Path(root).resolve();data=validate_frozen(root)
    bundle=root/'teacher_benchmark';legacy.initialize(bundle)
    manifest=[]
    for r in data:
        dest=bundle/'inputs/other'/f'{r["image_id"]}.png'
        if not dest.exists():shutil.copyfile(root/r['image_path'],dest)
    frame=legacy.inspect_inputs(bundle)
    registry=load_json(legacy_bundle/'model_registry.json')
    verified=legacy.verify_models(bundle,registry,keys)
    for spec in verified.values():
        if spec.get('kind')=='original':spec['benchmark_root']=str(root)
    legacy.run_benchmark(bundle,frame,verified,keys,require_gpu=True)
    byname={Path(r['image']).stem:r for r in frame.to_dict('records')}
    outputs={}
    for key in keys:
        outputs[key]={}
        for p in (bundle/'outputs'/key).glob('*/*/record.json'):
            record=load_json(p)
            if record.get('status')!='success':continue
            image_id=Path(record['image']).stem
            native_common=np.asarray(Image.open(p.parent/'common_ids.png'))
            converted=legacy_to_train(native_common)
            dest=root/'data/pseudo_labels'/key/f'{image_id}.png';dest.parent.mkdir(parents=True,exist_ok=True)
            Image.fromarray(converted).save(dest)
            Image.open(p.parent/'overlay.png').save(dest.with_name(image_id+'_overlay.png'))
            save_json(dest.with_suffix('.json'),{'image_id':image_id,'teacher':key,'record':record,
                'mask_sha':sha(dest),'pixels':{CLASSES[c]:int((converted==c).sum()) for c in range(9)},'verified':False})
            outputs[key][image_id]=converted
    priorities=[]
    for r in data:
        image_id=r['image_id']
        available=[outputs[k][image_id] for k in keys if image_id in outputs[k]]
        disagreement=None
        if len(available)==2:
            valid=(available[0]!=255)&(available[1]!=255)
            disagreement=float((available[0][valid]!=available[1][valid]).mean()) if valid.any() else None
            heat=np.zeros((*available[0].shape,3),np.uint8);heat[valid&(available[0]!=available[1])]=[255,0,0]
            Image.fromarray(heat).save(root/'data/review'/f'{image_id}_disagreement.png')
        priorities.append({'image_id':image_id,'split':r['split'],'disagreement_ratio':disagreement,'review_required':True})
    import pandas as pd
    pd.DataFrame(priorities).sort_values('disagreement_ratio',ascending=False).to_csv(root/'metrics/review_priority.csv',index=False)
    return priorities

def select_teacher(root):
    from .evaluation import confusion,metrics
    root=Path(root);data=validate_frozen(root);scores={}
    audit=[]
    for key in TEACHERS:
        cm=np.zeros((9,10),np.int64);n=0
        for r in data:
            if r['split'] not in ['train','validation'] or r['label_verified']!='true':continue
            pred=root/'data/pseudo_labels'/key/f'{r["image_id"]}.png'
            if not pred.exists():continue
            gt=read_mask(root/r['mask_path']);cm+=confusion(gt,read_mask(pred,gt.shape));n+=1
            if key==TEACHERS[0]:audit.append({'image_id':r['image_id'],'split':r['split'],'mask_sha':r['mask_sha']})
        values=metrics(cm)['per_class']
        supported=[v['iou'] for v in values[1:] if v['iou'] is not None]
        if not n or not supported:raise ValueError('Reviewed train/validation masks and both teacher predictions required')
        scores[key]={'macro_iou':float(np.mean(supported)),'n_images':n}
    # Require identical reviewed images for both Teachers, not unequal available subsets.
    ids=[{r['image_id'] for r in data if r['split'] in ['train','validation'] and r['label_verified']=='true'
          and (root/'data/pseudo_labels'/k/f'{r["image_id"]}.png').exists()} for k in TEACHERS]
    if ids[0]!=ids[1]:raise ValueError('Teacher comparison must use the same reviewed images')
    selected=max(TEACHERS,key=lambda k:scores[k]['macro_iou'])
    result={'selected':selected,'scores':scores,'reviewed_selection_samples':audit,'created_at':now()}
    save_json(root/'metrics/teacher_selection.json',result)
    for r in data:
        if r['split']=='train' and r['status'] not in ['reviewed','rejected']:
            p=root/'data/pseudo_labels'/selected/f'{r["image_id"]}.png'
            if p.exists():r.update(mask_path=str(p.relative_to(root)),mask_sha=sha(p),label_source=selected,
                                  label_verified='false',status='pseudo')
    write_rows(root,data);return result

def export_review(root,destination,teacher=TEACHERS[0]):
    root=Path(root);data=validate_frozen(root)
    with zipfile.ZipFile(destination,'w',zipfile.ZIP_DEFLATED) as z:
        z.writestr('classes.json',json.dumps({'classes':dict(enumerate(CLASSES)),'ignore':255,'palette':COLORS.tolist()},indent=2))
        z.writestr('README.txt','Edit masks externally. Return original-resolution integer ID PNG (0..8, ignore=255).\nApproval requires explicit reviewer. Drafts are not ground truth.\n')
        z.write(root/'data/dataset_metadata.csv','dataset_metadata.csv')
        for r in data:
            z.write(root/r['image_path'],'images/'+r['image_id']+'.png')
            for key in TEACHERS:
                for suffix,category in [('.png','draft_masks'),('_overlay.png','overlays')]:
                    p=root/'data/pseudo_labels'/key/(r['image_id']+suffix)
                    if p.exists():z.write(p,category+'/'+key+'/'+p.name)
    return str(destination)

def import_review_batch(root,folder,reviewer,confirmed=False,color_map=None):
    if not confirmed:raise ValueError('Explicit confirmed=True required after human review')
    known={r['image_id'] for r in rows(root)}
    files=sorted(Path(folder).glob('*.png'))
    if any(p.stem not in known for p in files):raise ValueError('Return masks only, named image_id.png; overlays and unknown filenames are not accepted')
    accepted=[]
    for p in files:
        accepted.append(import_mask(root,p.stem,p,reviewer,True,color_map,'manual_review'))
    if not accepted:raise ValueError('No review masks imported')
    return accepted

def import_aihub_polygons(root,image_id,json_path,label_mapping,reviewer='',verified=False):
    """Explicit adapter: normalize source to polygons[{label,points:[[x,y],...]}].
    Official package schemas vary; do not guess attribute combinations or RGB palettes.
    Unknown labels remain ignore, and official annotations are not automatically reviewed.
    """
    from PIL import ImageDraw
    row=next(r for r in rows(root) if r['image_id']==image_id)
    payload=load_json(json_path)
    if 'polygons' not in payload:raise ValueError('Normalize AI Hub JSON to polygons[{label,points}] using source guide')
    mask=Image.new('L',(int(row['image_width']),int(row['image_height'])),255);draw=ImageDraw.Draw(mask)
    for polygon in payload['polygons']:
        if polygon['label'] not in label_mapping:continue
        cid=int(label_mapping[polygon['label']])
        if cid not in range(9):raise ValueError('Explicit 9-class mapping required')
        points=[tuple(map(float,p)) for p in polygon['points']]
        if len(points)<3:raise ValueError('Polygon needs at least three points')
        draw.polygon(points,fill=cid)
    tmp=Path(root)/'data/review'/f'{image_id}_official.png';mask.save(tmp)
    return import_mask(root,image_id,tmp,reviewer,verified,label_source='AI Hub official polygons')
