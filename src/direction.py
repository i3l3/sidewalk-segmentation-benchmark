from pathlib import Path
import numpy as np
import pandas as pd
from scipy.ndimage import label,binary_dilation,distance_transform_edt
from .schema import *
from .dataset import validate_frozen,write_rows

DIRECTIONS=['Left','Straight','Right','No clear path']
FEATURES=['left_area_ratio','center_area_ratio','right_area_ratio','walkable_continuity',
          'path_width_ratio','center_offset','curb_proximity']
DEFAULT_RULE={'minimum_continuity':.4,'minimum_width':.05,'minimum_area':.03,'center_margin':.02}

def extract_features(mask,obstacle_mask=None):
    """Ratios refer only to image coordinates; no physical width/distance is inferred.
    ROI is lower 60% of image. Seed strip: bottom 5%, middle 20% of width.
    Keep the seed-connected component with the most seed pixels, tie by area.
    """
    h,w=mask.shape;roi_y=int(.4*h)
    walk=np.isin(mask,[2,5]);walk[:roi_y]=False
    if obstacle_mask is not None:
        if obstacle_mask.shape!=mask.shape:raise ValueError('Obstacle mask shape mismatch')
        walk &= ~obstacle_mask.astype(bool)
    components,n=label(walk,structure=np.ones((3,3)))
    seed=np.zeros_like(walk);seed[max(roi_y,int(.95*h)):h,int(.4*w):max(int(.4*w)+1,int(.6*w))]=True
    ids,counts=np.unique(components[seed & walk],return_counts=True)
    if not len(ids):connected=np.zeros_like(walk)
    else:
        chosen=max(zip(ids,counts),key=lambda v:(v[1],int((components==v[0]).sum())))[0]
        connected=components==chosen
    roi=connected[roi_y:];bands=np.array_split(roi,3,axis=1)
    areas=[float(b.mean()) if b.size else 0. for b in bands]
    occupied=roi.any(1);continuity=float(occupied.mean()) if len(occupied) else 0.
    # Width means longest horizontal connected run in each occupied ROI row / image width.
    widths=[]
    for line in roi[occupied]:
        changes=np.diff(np.r_[False,line,False].astype(int));widths.extend([float(max(np.where(changes==-1)[0]-np.where(changes==1)[0]))/w])
    width=float(np.median(widths)) if widths else 0.
    yy,xx=np.where(roi)
    offset=float((xx.mean()/max(w-1,1)-.5)*2) if len(xx) else 0.
    curb=mask==3
    proximity=float(max(0,1-distance_transform_edt(~curb)[connected].min()/np.hypot(h,w))) if curb.any() and connected.any() else 0.
    result=dict(zip(FEATURES,areas+[continuity,width,offset,proximity]))
    result['obstacle_ratio']=float(obstacle_mask[roi_y:].mean()) if obstacle_mask is not None else None
    result['has_seed_connection']=bool(connected.any())
    return result,connected

def rule_direction(features,config=None):
    c={**DEFAULT_RULE,**(config or {})}
    if not features['has_seed_connection'] or features['walkable_continuity']<c['minimum_continuity'] or features['path_width_ratio']<c['minimum_width']:
        return 'No clear path'
    scores=[features[k] for k in FEATURES[:3]]
    if max(scores)<c['minimum_area']:return 'No clear path'
    if scores[1]>=max(scores)-c['center_margin']:return 'Straight'
    return DIRECTIONS[int(np.argmax(scores))]

def set_direction_label(root,image_id,direction,reviewer,confirmed=False):
    if direction not in DIRECTIONS or not reviewer.strip() or not confirmed:raise ValueError('Human direction label and explicit reviewer confirmation required')
    data=validate_frozen(root);row=next(r for r in data if r['image_id']==image_id)
    row.update(direction_label=direction,direction_verified='true',direction_reviewer=reviewer,
               direction_reviewed_at=now(),direction_image_sha=row['image_sha']);write_rows(root,data)

def feature_table(root,model,prediction_dir):
    root=Path(root);records=[]
    for row in validate_frozen(root):
        path=Path(prediction_dir)/f'{row["image_id"]}.png'
        if not path.exists():continue
        mask=read_mask(path,(int(row['image_height']),int(row['image_width'])))
        features,connected=extract_features(mask)
        out=root/'outputs/direction_predictions'/model;out.mkdir(parents=True,exist_ok=True)
        from PIL import Image
        Image.fromarray((connected*255).astype(np.uint8)).save(out/f'{row["image_id"]}_connected.png')
        records.append({'model':model,'image_id':row['image_id'],'split':row['split'],'group_id':row['group_id'],
                        **features,'direction_label':row['direction_label'] if row['direction_verified']=='true' else '',
                        'direction_verified':row['direction_verified'],'rule_prediction':rule_direction(features)})
    frame=pd.DataFrame(records)
    if not frame.empty:frame.to_csv(root/'metrics'/f'{model}_direction_features.csv',index=False)
    return frame

def freeze_rule(root,features):
    from sklearn.metrics import f1_score
    root=Path(root);path=root/'data/splits/direction_rule.json'
    val=features[(features.split=='validation')&(features.direction_verified.astype(str).str.lower()=='true')]
    labels=sorted((str(r['image_id']),str(r['direction_label'])) for r in val.to_dict('records'))
    if path.exists():
        existing=load_json(path)
        if existing.get('validation_labels')!=[list(v) for v in labels]:raise ValueError('Direction validation labels changed; start a new rule version before final testing')
        return existing
    best=dict(DEFAULT_RULE);best_score=None
    if not val.empty:
        for continuity in [.3,.4,.5]:
            for width in [.03,.05,.1]:
                cfg={**DEFAULT_RULE,'minimum_continuity':continuity,'minimum_width':width}
                preds=[rule_direction(r,cfg) for r in val.to_dict('records')]
                score=f1_score(val.direction_label,preds,labels=DIRECTIONS,average='macro',zero_division=0)
                if best_score is None or score>best_score:best_score=float(score);best=cfg
    result={'config':best,'validation_macro_f1':best_score,'reason':'validation_tuned' if best_score is not None else 'predeclared_defaults_no_validation_labels',
            'tuned_image_ids':val.image_id.tolist(),'validation_labels':[list(v) for v in labels],'created_at':now()}
    save_json(path,result);return result

def classify(root,features,tag):
    """Train a separate classifier per segmentation model: no cross-model feature mixing."""
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import classification_report,confusion_matrix,accuracy_score,f1_score
    import joblib
    root=Path(root);out=root/'metrics/direction'/tag;out.mkdir(parents=True,exist_ok=True)
    if features.empty:
        save_json(out/'status.json',{'trained':False,'reason':'No segmentation features'});return
    frozen=freeze_rule(root,features)
    features=features.copy()
    features['rule_prediction']=[rule_direction(r,frozen['config']) for r in features.to_dict('records')]
    features.to_csv(root/'metrics'/f'{tag}_direction_features.csv',index=False)
    tables=[pd.read_csv(p) for p in (root/'metrics').glob('*_direction_features.csv')]
    if tables:pd.concat(tables,ignore_index=True).to_csv(root/'metrics/direction_features.csv',index=False)
    data=features[features.direction_verified.astype(str).str.lower()=='true'].copy()
    split={s:data[data.split==s] for s in ['train','validation','test']}
    counts={s:{c:int((split[s].direction_label==c).sum()) for c in DIRECTIONS} for s in split}
    enough=all(counts[s][c]>=(20 if s=='train' else 5) for s in counts for c in DIRECTIONS)
    models={'rule':None}
    if enough:
        candidates={'decision_tree':[DecisionTreeClassifier(max_depth=d,min_samples_leaf=3,random_state=42) for d in [3,5,8]],
                    'random_forest':[RandomForestClassifier(n_estimators=200,max_depth=d,min_samples_leaf=2,random_state=42,n_jobs=-1) for d in [5,10,None]]}
        for name,choices in candidates.items():
            scored=[]
            for estimator in choices:
                estimator.fit(split['train'][FEATURES],split['train'].direction_label)
                predicted=estimator.predict(split['validation'][FEATURES])
                score=f1_score(split['validation'].direction_label,predicted,labels=DIRECTIONS,average='macro',zero_division=0)
                scored.append((score,estimator))
            models[name]=max(scored,key=lambda x:x[0])[1];joblib.dump(models[name],out/f'{name}.joblib')
    save_json(out/'status.json',{'trained':enough,'counts':counts,'reason':None if enough else 'Need >=20 train / 5 validation / 5 test per direction','feature_model':tag})
    test=split['test']
    for name,estimator in (models.items() if not test.empty else []):
        prediction=estimator.predict(test[FEATURES]) if estimator else [rule_direction(r,frozen['config']) for r in test.to_dict('records')]
        save_json(out/f'{name}_metrics.json',{'accuracy':accuracy_score(test.direction_label,prediction),
            'macro_f1':f1_score(test.direction_label,prediction,labels=DIRECTIONS,average='macro',zero_division=0),
            'report':classification_report(test.direction_label,prediction,labels=DIRECTIONS,output_dict=True,zero_division=0),
            'confusion_matrix':confusion_matrix(test.direction_label,prediction,labels=DIRECTIONS).tolist(),'labels':DIRECTIONS})
        pd.DataFrame({'image_id':test.image_id,'label':test.direction_label,'prediction':prediction}).to_csv(out/f'{name}_predictions.csv',index=False)
        import matplotlib;matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from sklearn.metrics import ConfusionMatrixDisplay
        fig,ax=plt.subplots(figsize=(7,6));ConfusionMatrixDisplay.from_predictions(test.direction_label,prediction,labels=DIRECTIONS,ax=ax,xticks_rotation=30)
        fig.tight_layout();fig.savefig(out/f'{name}_confusion.png',dpi=160);plt.close(fig)
    # Render rule candidates for all splits, including images without direction GT.
    from PIL import Image,ImageDraw
    from .dataset import rows
    lookup={r['image_id']:r for r in rows(root)}
    visual=root/'outputs/direction_predictions'/tag;visual.mkdir(parents=True,exist_ok=True)
    for r in features.to_dict('records'):
        if r['image_id'] not in lookup:continue
        image=Image.open(root/lookup[r['image_id']]['image_path']).convert('RGB')
        draw=ImageDraw.Draw(image);w,h=image.size;candidate=r['rule_prediction']
        draw.rectangle((0,0,w,min(40,h)),fill='white');draw.text((8,10),'Direction candidate: '+candidate,fill='black')
        if candidate!='No clear path':
            start=(w//2,int(.9*h));dx={'Left':-.2,'Straight':0,'Right':.2}[candidate]
            end=(int(w*(.5+dx)),int(.6*h));draw.line((start,end),fill='lime',width=max(2,w//100))
            draw.ellipse((end[0]-5,end[1]-5,end[0]+5,end[1]+5),fill='lime')
        image.save(visual/f'{r["image_id"]}_direction.jpg',quality=92)
