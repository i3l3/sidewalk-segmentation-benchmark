from pathlib import Path
import time
import numpy as np
import pandas as pd
from PIL import Image,ImageDraw,ImageOps
from scipy.ndimage import binary_erosion,binary_dilation
from .schema import *
from .dataset import validate_frozen

def confusion(gt,pred):
    valid=gt!=255
    # Unknown predictions remain errors, rather than removing those GT pixels.
    predicted=np.where(pred[valid]==255,9,pred[valid])
    return np.bincount((gt[valid].astype(np.int64)*10+predicted),minlength=90).reshape(9,10)

def metrics(cm,support=None):
    support=set(range(9)) if support is None else set(support);per=[]
    for c,name in enumerate(CLASSES):
        tp=int(cm[c,c]);actual=int(cm[c].sum());predicted=int(cm[:,c].sum());union=actual+predicted-tp
        per.append({'class_id':c,'class_name':name,'supported':c in support,'gt_pixels':actual,
                    'iou':tp/union if union and c in support else None,
                    'precision':tp/predicted if predicted and c in support else None,
                    'recall':tp/actual if actual and c in support else None,
                    'missing_reason':'unsupported_class' if c not in support else ('class_absent_in_gt_and_prediction' if not union else None),
                    'precision_missing_reason':'no_predicted_pixels' if not predicted else None,
                    'recall_missing_reason':'no_gt_pixels' if not actual else None})
    values=[r['iou'] for r in per if r['iou'] is not None]
    return {'miou':float(np.mean(values)) if values else None,
            'pixel_accuracy':float(np.trace(cm)/cm.sum()) if cm.sum() else None,'per_class':per}

def boundary_counts(gt,pred,c,tolerance=2):
    valid=gt!=255
    safe=valid & ~binary_dilation(~valid,iterations=tolerance+1)
    edge=lambda x:(x&~binary_erosion(x,border_value=1))&safe
    a=edge(gt==c);b=edge(pred==c)
    return np.array([(b&binary_dilation(a,iterations=tolerance)).sum(),b.sum(),
                     (a&binary_dilation(b,iterations=tolerance)).sum(),a.sum()],dtype=np.int64)

def boundary_f1(counts):
    mp,npred,mg,ngt=counts
    if not npred and not ngt:return None
    p=mp/npred if npred else 0;r=mg/ngt if ngt else 0
    return float(2*p*r/(p+r)) if p+r else 0.

def color_mask(mask):
    color=np.zeros((*mask.shape,3),np.uint8);valid=mask!=255;color[valid]=COLORS[mask[valid]]
    return Image.fromarray(color)

def visualize(root,row,pred,model):
    root=Path(root);dest=root/'outputs/comparisons'/model;dest.mkdir(parents=True,exist_ok=True)
    image=Image.open(root/row['image_path']).convert('RGB');gt=read_mask(root/row['mask_path'],pred.shape)
    valid=gt!=255;error=np.asarray(image).copy();error[valid&(gt!=pred)]=[255,0,0]
    panes=[image,color_mask(gt),color_mask(pred),Image.fromarray(error)]
    titles=['Original','Reviewed GT','Prediction','Errors (red)']
    canvas=Image.new('RGB',(1200,640),'white');draw=ImageDraw.Draw(canvas)
    for i,(pane,title) in enumerate(zip(panes,titles)):
        pane=pane.copy();pane.thumbnail((600,280));x=(i%2)*600;y=(i//2)*320
        canvas.paste(pane,(x,y+30));draw.text((x+8,y+8),title,fill='black')
    canvas.save(dest/f'{row["image_id"]}.jpg',quality=93)
    # Same GT-derived crop for every model: never select a flattering prediction crop.
    yy,xx=np.where(gt==3)
    if len(xx):
        cx=int(np.median(xx));cy=int(np.median(yy));w=max(64,image.width//3);h=max(64,image.height//3)
        box=(max(0,cx-w//2),max(0,cy-h//2),min(image.width,cx+w//2),min(image.height,cy+h//2))
        zoom=Image.new('RGB',(1200,400),'white')
        for i,pane in enumerate(panes[:3]):
            part=pane.crop(box);part.thumbnail((400,370));zoom.paste(part,(i*400,30))
            ImageDraw.Draw(zoom).text((i*400+8,8),titles[i],fill='black')
        zoom.save(dest/f'{row["image_id"]}_curb.jpg',quality=93)

def evaluate_predictions(root,model,prediction_dir,support=range(9),split='test',run_info=None):
    root=Path(root);data=validate_frozen(root)
    selected=[r for r in data if r['split']==split]
    if not selected or any(r['label_verified']!='true' for r in selected):
        raise ValueError(f'All {split} images require human-reviewed labels before evaluation')
    groups={};records=[];classes=[]
    for row in selected:
        pred=read_mask(Path(prediction_dir)/f'{row["image_id"]}.png')
        gt=read_mask(root/row['mask_path'],pred.shape);cm=confusion(gt,pred);m=metrics(cm,support)
        if not cm.sum():raise ValueError('No valid evaluation pixels')
        b=[boundary_counts(gt,pred,c) for c in range(9)]
        average=[boundary_f1(b[c]) for c in support]
        average=[v for v in average if v is not None]
        stats=(run_info or {}).get(row['image_id'],{})
        byclass={v['class_id']:v for v in m['per_class']}
        records.append({'model':model,'image_id':row['image_id'],'split':split,
                        'sidewalk_iou':byclass[2]['iou'],'curb_iou':byclass[3]['iou'],
                        'crosswalk_iou':byclass[4]['iou'],'boundary_f1':float(np.mean(average)) if average else None,
                        'pixel_accuracy':m['pixel_accuracy'],'miou':m['miou'],
                        'inference_seconds':stats.get('seconds'),'peak_gpu_mb':stats.get('peak_gpu_mb'),
                        'missing_reason':'unsupported_class' if set(support)!=set(range(9)) else ''})
        for v in m['per_class']:classes.append({'model':model,'image_id':row['image_id'],**v,
                    'boundary_f1':boundary_f1(b[v['class_id']]) if v['supported'] else None})
        group_names=['all']
        if row['camera_view']=='pedestrian':group_names.append('korean_pedestrian' if row['country'].lower() in ['korea','kr'] else 'non_korean_pedestrian')
        for group in group_names:
            if group not in groups:groups[group]={'cm':np.zeros((9,10),np.int64),'b':np.zeros((9,4),np.int64),'n':0}
            groups[group]['cm']+=cm;groups[group]['b']+=b;groups[group]['n']+=1
        visualize(root,row,pred,model)
    summaries=[]
    for group,values in groups.items():
        m=metrics(values['cm'],support);common=metrics(values['cm'],[0,1,2])
        summaries.append({'model':model,'group':group,'n':values['n'],'miou':m['miou'],
                          'common_other_road_sidewalk_miou':common['miou'],'pixel_accuracy':m['pixel_accuracy']})
        for c in m['per_class']:c['boundary_f1']=boundary_f1(values['b'][c['class_id']]) if c['supported'] else None
        save_json(root/'metrics'/model/f'{group}_aggregate.json',m)
    score={r['group']:r for r in summaries}
    if all(g in score and score[g]['common_other_road_sidewalk_miou'] is not None for g in ['korean_pedestrian','non_korean_pedestrian']):
        weighted=(2*score['non_korean_pedestrian']['common_other_road_sidewalk_miou']+score['korean_pedestrian']['common_other_road_sidewalk_miou'])/3
    else:weighted=None
    save_json(root/'metrics'/model/'weighted_score.json',{'score':weighted,'definition':'(2*non-Korean pedestrian common mIoU + Korean pedestrian common mIoU)/3','missing_reason':None if weighted is not None else 'Both disjoint groups required'})
    folder=root/'metrics'/model;folder.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(records).to_csv(folder/'segmentation_metrics.csv',index=False)
    pd.DataFrame(classes).to_csv(folder/'class_metrics.csv',index=False)
    pd.DataFrame(summaries).to_csv(folder/'summary.csv',index=False)
    return summaries

def plots(root):
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    root=Path(root);out=root/'reports/graphs';out.mkdir(parents=True,exist_ok=True)
    paths=list((root/'metrics').glob('*/segmentation_metrics.csv'))
    if paths:
        frame=pd.concat([pd.read_csv(p) for p in paths],ignore_index=True)
        frame.to_csv(root/'metrics/segmentation_metrics.csv',index=False)
        for metric in ['miou','boundary_f1','inference_seconds','peak_gpu_mb']:
            values=frame.groupby('model')[metric].mean().dropna()
            if len(values):
                fig,ax=plt.subplots(figsize=(10,5));values.plot.bar(ax=ax);ax.set_ylabel(metric)
                fig.tight_layout();fig.savefig(out/f'{metric}.png',dpi=160);plt.close(fig)
    history_paths=list((root/'checkpoints').glob('*/history.json'))
    for p in history_paths:
        history=load_json(p)
        if not history:continue
        frame=pd.DataFrame(history);fig,ax=plt.subplots(figsize=(8,4))
        ax.plot(frame.epoch,frame.train_loss,label='train');ax.plot(frame.epoch,frame.val_loss,label='validation');ax.legend()
        fig.tight_layout();fig.savefig(out/f'{p.parent.name}_loss.png',dpi=160);plt.close(fig)
        fig,ax=plt.subplots(figsize=(8,4));ax.plot(frame.epoch,frame.val_miou);ax.set_ylabel('Validation mIoU')
        fig.tight_layout();fig.savefig(out/f'{p.parent.name}_validation_miou.png',dpi=160);plt.close(fig)
    class_paths=list((root/'metrics').glob('*/class_metrics.csv'))
    if class_paths:
        df=pd.concat([pd.read_csv(p) for p in class_paths]);table=df.pivot_table(index='class_name',columns='model',values='iou',aggfunc='mean')
        if not table.empty:
            ax=table.plot.bar(figsize=(12,6));fig=ax.get_figure();fig.tight_layout();fig.savefig(out/'class_iou.png',dpi=160);plt.close(fig)
