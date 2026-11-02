from pathlib import Path
import csv
import hashlib
import random
import shutil
import zipfile
import numpy as np
from PIL import Image,ImageOps
from .schema import init,sha,load_json,save_json,now,read_mask

FIELDS=['image_id','source','camera_view','country','image_width','image_height','split',
        'label_source','label_verified','dominant_class','group_id','image_path','image_sha',
        'pixel_sha','dhash','mask_path','mask_sha','reviewer','reviewed_at','status','direction_label',
        'direction_verified','direction_reviewer','direction_reviewed_at','direction_image_sha','license','original_path','group_hint']

def rows(root):
    path=Path(root)/'data/dataset_metadata.csv'
    if not path.exists():return []
    with path.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))

def write_rows(root,data):
    path=Path(root)/'data/dataset_metadata.csv';path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=FIELDS);w.writeheader();w.writerows({k:r.get(k,'') for k in FIELDS} for r in data)

def extract_zip(source,target):
    target=Path(target).resolve();target.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(source) as z:
        for info in z.infolist():
            resolved=(target/info.filename).resolve()
            if not resolved.is_relative_to(target) or (info.external_attr>>16)&0o170000==0o120000:
                raise ValueError('Unsafe archive entry')
        z.extractall(target)

def import_images(root,source,source_name='AI Hub sidewalk (dataset 189)',country='korea',license_note='User must confirm dataset terms',max_images=350):
    root=init(root);source=Path(source).resolve()
    if (root/'data/splits/locked.json').exists():raise ValueError('Dataset is frozen; import into a new dataset version')
    if source.suffix.lower()=='.zip':
        unpack=root/'data/import_cache'/sha(source)[:16];extract_zip(source,unpack);source=unpack
    if not source.is_dir():raise ValueError('Image folder or ZIP required')
    existing=rows(root);known={r['image_sha'] for r in existing}
    exclusions=load_json(root/'data/excluded_hashes.json') if (root/'data/excluded_hashes.json').exists() else []
    errors=[]
    candidates=sorted(p for p in source.rglob('*') if p.is_file() and p.suffix.lower() in {'.jpg','.jpeg','.png','.webp','.bmp'})
    random.Random(42).shuffle(candidates)
    for p in candidates:
        if max_images and len(existing)>=max_images:break
        if not p.is_file() or p.suffix.lower() not in {'.jpg','.jpeg','.png','.webp','.bmp'}:continue
        digest=sha(p)
        if digest in known or digest in exclusions:continue
        try:
            with Image.open(p) as im:im=ImageOps.exif_transpose(im).convert('RGB');im.load()
            pixel=hashlib.sha256(im.tobytes()+str(im.size).encode()).hexdigest()
            if pixel in exclusions:continue
            if pixel in {r['pixel_sha'] for r in existing}:continue
            small=np.asarray(im.convert('L').resize((9,8),Image.Resampling.BILINEAR))
            bits=(small[:,1:]>small[:,:-1]).flatten()
            dhash=f'{sum(int(x)<<i for i,x in enumerate(bits)):016x}'
            image_id='im_'+pixel[:20];dest=root/'data/raw'/f'{image_id}.png';im.save(dest)
            relative=p.relative_to(source)
            # Parent is a suggestion only. Split is blocked until explicit group confirmation.
            existing.append(dict(zip(FIELDS,['']*len(FIELDS)))|{'image_id':image_id,'source':source_name,
                'camera_view':'unknown','country':country,'image_width':im.width,'image_height':im.height,
                'image_path':str(dest.relative_to(root)),'image_sha':sha(dest),'pixel_sha':pixel,'dhash':dhash,
                'group_id':'','label_verified':'false','status':'unlabeled','direction_verified':'false',
                'license':license_note,'original_path':relative.as_posix(),'group_hint':relative.parent.as_posix()})
            known.add(digest)
        except Exception as e:errors.append({'path':str(p),'error':str(e)})
    write_rows(root,existing);save_json(root/'metrics/import_errors.json',errors)
    return existing

def refresh_reviews(root,data=None):
    data=rows(root) if data is None else data
    changed=False
    for r in data:
        image=Path(root)/r['image_path'];mask=Path(root)/r['mask_path'] if r['mask_path'] else None
        valid_image=image.exists() and sha(image)==r['image_sha']
        if r['label_verified']=='true' and (not r['reviewer'].strip() or not r['reviewed_at'] or r['status']!='reviewed' or not valid_image or not mask or not mask.exists() or sha(mask)!=r['mask_sha']):
            r.update(label_verified='false',status='needs_review');changed=True
        if r['direction_verified']=='true' and (not valid_image or r['direction_image_sha']!=r['image_sha']):
            r['direction_verified']='false';changed=True
    if changed:write_rows(root,data)
    return data

def split_dataset(root,seed=42,targets=(250,50,50),confirm_groups=False):
    root=Path(root);data=refresh_reviews(root)
    if not data:raise ValueError('No images imported')
    lock=root/'data/splits/locked.json'
    if lock.exists():
        frozen=load_json(lock)
        current={r['image_id']:{'image_sha':r['image_sha'],'group_id':r['group_id']} for r in data}
        if current!=frozen['images']:raise ValueError('Frozen dataset membership/group changed; create a new dataset version')
        expected={i:s for s,ids in frozen['splits'].items() for i in ids}
        if any(r['split']!=expected[r['image_id']] for r in data):raise ValueError('Frozen split was edited')
        return frozen
    if not confirm_groups or any(not r['group_id'].strip() for r in data):
        raise ValueError('Fill group_id by place/sequence and explicitly confirm_groups=True')
    # Union visually similar images into one split group; never silently discard across splits.
    parent=list(range(len(data)))
    def find(i):
        while parent[i]!=i:parent[i]=parent[parent[i]];i=parent[i]
        return i
    def union(i,j):parent[find(i)]=find(j)
    pairs=[]
    for i,a in enumerate(data):
        for j in range(i):
            b=data[j];distance=(int(a['dhash'],16)^int(b['dhash'],16)).bit_count()
            if a['group_id']==b['group_id'] or a['pixel_sha']==b['pixel_sha'] or distance<=4:
                union(i,j)
                if a['group_id']!=b['group_id']:pairs.append({'a':a['image_id'],'b':b['image_id'],'dhash_distance':distance})
    groups={}
    for i,r in enumerate(data):groups.setdefault(find(i),[]).append(r)
    values=list(groups.values());random.Random(seed).shuffle(values);values.sort(key=len,reverse=True)
    ratios=np.asarray(targets,dtype=float)/sum(targets);desired=ratios*len(data);counts=np.zeros(3,int)
    names=['train','validation','test']
    if len(values)<3:raise ValueError('At least three independent scene groups required')
    for group in values:
        k=int(np.argmax((desired-counts)/np.maximum(desired,1)))
        for r in group:r['split']=names[k]
        counts[k]+=len(group)
    if min(counts)==0:raise ValueError('Cannot create nonempty grouped splits')
    write_rows(root,data)
    result={'seed':seed,'targets':dict(zip(names,targets)),'actual':dict(zip(names,map(int,counts))),
            'images':{r['image_id']:{'image_sha':r['image_sha'],'group_id':r['group_id']} for r in data},
            'splits':{s:[r['image_id'] for r in data if r['split']==s] for s in names},'created_at':now()}
    save_json(lock,result);save_json(root/'metrics/similar_image_pairs.json',pairs)
    return result

def validate_frozen(root):
    root=Path(root);data=refresh_reviews(root);lock=load_json(root/'data/splits/locked.json')
    expected={i:s for s,ids in lock['splits'].items() for i in ids}
    if set(expected)!={r['image_id'] for r in data}:raise ValueError('Dataset membership changed')
    groups={}
    for r in data:
        if r['split']!=expected[r['image_id']] or r['image_sha']!=lock['images'][r['image_id']]['image_sha'] or r['group_id']!=lock['images'][r['image_id']]['group_id']:
            raise ValueError('Frozen split/image/group modified')
        if sha(root/r['image_path'])!=r['image_sha']:raise ValueError('Image bytes changed')
        groups.setdefault(r['group_id'],set()).add(r['split'])
    if any(len(v)>1 for v in groups.values()):raise ValueError('Group leakage')
    return data
