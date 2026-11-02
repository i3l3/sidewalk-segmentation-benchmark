from pathlib import Path
import hashlib
import json
from datetime import datetime, timezone
import numpy as np
from PIL import Image

CLASSES = ['Other','Road','Sidewalk','Curb','Crosswalk (plain)','Pedestrian Area',
           'Curb Cut','Crosswalk Marking','Bike Lane']
IGNORE = 255
LEGACY_TO_TRAIN = {0:0,1:1,2:2,3:3,4:4,7:5,8:6,9:7,10:8}
COLORS = np.array([[65,65,65],[100,110,150],[225,95,170],[255,155,45],
                   [60,220,170],[150,90,225],[255,185,130],[235,245,245],[50,160,255]],dtype=np.uint8)

def now(): return datetime.now(timezone.utc).isoformat()

def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for part in iter(lambda:f.read(1<<20),b''):h.update(part)
    return h.hexdigest()

def save_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2,default=str),encoding='utf-8');tmp.replace(path)

def load_json(path): return json.loads(Path(path).read_text(encoding='utf-8'))

def legacy_to_train(mask):
    mask=np.asarray(mask)
    bad=set(np.unique(mask))-(set(range(12))|{65535})
    if bad: raise ValueError(f'Unknown legacy classes: {bad}')
    result=np.full(mask.shape,IGNORE,dtype=np.uint8)
    for old,new in LEGACY_TO_TRAIN.items():result[mask==old]=new
    return result

def read_mask(path,shape=None):
    a=np.asarray(Image.open(path))
    if a.ndim!=2 or a.dtype.kind not in 'ui':raise ValueError('Expected single-channel integer ID PNG')
    if shape is not None and a.shape!=tuple(shape):raise ValueError(f'Mask shape {a.shape} != {shape}')
    if set(np.unique(a))-(set(range(9))|{IGNORE}):raise ValueError('Invalid training IDs (0..8,255 only)')
    return a.astype(np.uint8)

def init(root):
    root=Path(root).resolve()
    for folder in ['data/raw','data/pseudo_labels','data/verified_labels','data/splits',
                   'data/review','metrics','reports','checkpoints','outputs','logs']:
        (root/folder).mkdir(parents=True,exist_ok=True)
    save_json(root/'data/class_schema.json',{'classes':dict(enumerate(CLASSES)),'ignore_index':255,
        'legacy_to_train':LEGACY_TO_TRAIN,'excluded_legacy_ids':[5,6,11],
        'notes':'Tactile paving / stairs / caution zone excluded from phase 2 ontology; not Other.'})
    return root
