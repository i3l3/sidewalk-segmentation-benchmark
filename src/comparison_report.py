"""Reproducible A/C comparison after fixed-test evaluation has completed."""


def build_report(root):
    from pathlib import Path
    import json,csv,hashlib,math
    import numpy as np
    import pandas as pd
    from PIL import Image,ImageDraw
    import matplotlib.pyplot as plt
    from IPython.display import display
    import torch
    from safetensors.torch import load_file
    ROOT=Path(root)
    state=json.loads((ROOT/'logs/continuation_AC_status.json').read_text())
    assert state['state']=='completed',state
    lock=json.loads((ROOT/'data/splits/locked.json').read_text())
    test_ids=lock['splits']['test'];assert len(test_ids)==60
    with (ROOT/'data/dataset_metadata.csv').open(encoding='utf-8-sig') as stream:data=list(csv.DictReader(stream))
    lookup={r['image_id']:r for r in data}
    assert all(lookup[i]['split']=='test' and lookup[i]['label_verified']=='true' for i in test_ids)
    ids=[1,2,8];labels=['Road (incl. alley)','Sidewalk','Bike Lane']
    summary=[];per_class=[];histories={};aggregates={}
    def digest(path):
        h=hashlib.sha256()
        with path.open('rb') as stream:
            while chunk:=stream.read(1024*1024):h.update(chunk)
        return h.hexdigest()
    for key,name in [('A','B0'),('C','B2')]:
        folder=ROOT/'checkpoints'/key
        history=json.loads((folder/'history.json').read_text());histories[key]=history
        run=json.loads((folder/'run.json').read_text());assert run['config']['class_ids']==ids and not run['tiny_synthetic_only']
        checkpoint=torch.load(folder/'latest.pt',map_location='cpu',weights_only=False)
        assert checkpoint['signature']==run['signature'] and checkpoint['epoch']==history[-1]['epoch']-1
        best=max(history,key=lambda row:row['val_miou']);assert abs(best['val_miou']-checkpoint['best'])<1e-12
        weights=load_file(str(folder/'best/model.safetensors'));assert weights['decode_head.classifier.weight'].shape[0]==3
        assert all(torch.isfinite(t).all().item() for t in weights.values() if t.is_floating_point())
        parameters=sum(t.numel() for key,t in weights.items() if not any(token in key for token in ['running_mean','running_var','num_batches_tracked']))
        del checkpoint,weights
        prediction_ids={p.stem for p in (ROOT/'outputs/masks'/key).glob('*.png')}
        assert prediction_ids==set(test_ids),(key,'prediction mismatch')
        aggregate=json.loads((ROOT/'metrics'/key/'all_aggregate.json').read_text());aggregates[key]=aggregate
        frame=pd.read_csv(ROOT/'metrics'/key/'segmentation_metrics.csv');assert set(frame.image_id)==set(test_ids)
        assert all(math.isfinite(v) for v in [aggregate['miou'],aggregate['pixel_accuracy']])
        row={'experiment':key,'model':name,'epochs_completed':len(history),'best_epoch':best['epoch'],
             'best_validation_miou':best['val_miou'],'test_miou':aggregate['miou'],'test_pixel_accuracy':aggregate['pixel_accuracy'],
             'test_images':len(test_ids),'mean_boundary_f1':float(frame.boundary_f1.mean()),
             'mean_inference_seconds':float(frame.inference_seconds.mean()),'max_inference_gpu_mib':float(frame.peak_gpu_mb.max()),
             'max_training_gpu_mib':max(h['peak_gpu_mb'] for h in history),'training_epoch_seconds':sum(h['epoch_seconds'] for h in history),
             'model_parameters':parameters,'best_weights_sha256':digest(folder/'best/model.safetensors'),
             'best_checkpoint':str(folder/'best'),'source_pretrained_model':run['origin']['model_id'],
             'pretrained_revision':run['origin']['revision']}
        summary.append(row)
        for item in aggregate['per_class']:
            if item['class_id'] in ids:per_class.append({'experiment':key,'model':name,**item})
    assert [r['gt_pixels'] for r in aggregates['A']['per_class']]==[r['gt_pixels'] for r in aggregates['C']['per_class']]
    reports=ROOT/'reports';graphs=reports/'graphs';graphs.mkdir(exist_ok=True,parents=True)
    summary_frame=pd.DataFrame(summary);class_frame=pd.DataFrame(per_class)
    summary_frame.to_csv(reports/'B0_B2_test_summary.csv',index=False)
    class_frame.to_csv(reports/'B0_B2_test_class_metrics.csv',index=False)
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for row in summary:
        key=row['experiment'];axes[0].plot([h['epoch'] for h in histories[key]],[h['val_miou']*100 for h in histories[key]],label=row['model'])
    axes[0].set(xlabel='Epoch',ylabel='Validation mIoU (%)',title='Checkpoint selection used validation only');axes[0].legend()
    x=np.arange(3);width=.35
    for offset,key in enumerate(['A','C']):
        values=[next(r['iou'] for r in aggregates[key]['per_class'] if r['class_id']==cid)*100 for cid in ids]
        axes[1].bar(x+(offset-.5)*width,values,width,label='B0' if key=='A' else 'B2')
    axes[1].set_xticks(x,labels);axes[1].set(ylabel='Test IoU (%)',title='Same fixed 60-image test set',ylim=(0,100));axes[1].legend()
    for ax in axes:ax.grid(axis='y',alpha=.2)
    fig.tight_layout();fig.savefig(graphs/'B0_B2_comparison.png',dpi=160);display(fig);plt.close(fig)
    # Deterministic visual examples selected from fixed test-list order, before examining predictions.
    examples=test_ids[:6];palette={1:(255,85,0),2:(225,95,170),8:(50,160,255)}
    canvas=Image.new('RGB',(1536,len(examples)*260+30),'white');draw=ImageDraw.Draw(canvas)
    for x,title in enumerate(['Original','Reviewed reference','B0 prediction','B2 prediction']):draw.text((x*384+6,8),title,fill='black')
    for y,image_id in enumerate(examples):
        row=lookup[image_id]
        with Image.open(ROOT/row['image_path']) as source:original=source.convert('RGB')
        with Image.open(ROOT/row['mask_path']) as source:gt=np.asarray(source)
        panes=[original]
        for mask_path in [ROOT/row['mask_path'],ROOT/'outputs/masks/A'/(image_id+'.png'),ROOT/'outputs/masks/C'/(image_id+'.png')]:
            with Image.open(mask_path) as source:mask=np.asarray(source)
            colors=np.full((*mask.shape,3),75,np.uint8)
            for cid,color in palette.items():colors[(mask==cid)&(gt!=255)]=color
            pane=Image.blend(original,Image.fromarray(colors),.65);panes.append(pane)
        for x,pane in enumerate(panes):
            pane.thumbnail((384,230));canvas.paste(pane,(x*384,30+y*260))
        draw.text((8,30+y*260+235),row['original_path'],fill='black')
    preview=reports/'B0_B2_test_examples.jpg';canvas.save(preview,quality=90)
    lines=['# B0 / B2 검수 라벨 실험 결과','',
           '학습 251장 / 검증 39장 / 고정 테스트 60장. 차도(골목 포함)·인도·자전거도로 3종.',
           'ImageNet 사전학습 인코더에 새 3종 헤드를 붙여 학습했다. 체크포인트는 검증 mIoU로 선택했으며 테스트 결과로 설정을 조정하지 않았다.','',
           '| 모델 | Best epoch | 검증 mIoU | 테스트 mIoU | 테스트 Pixel Accuracy |',
           '|---|---:|---:|---:|---:|']
    for row in summary:lines.append(f"| {row['model']} | {row['best_epoch']} | {row['best_validation_miou']*100:.2f}% | {row['test_miou']*100:.2f}% | {row['test_pixel_accuracy']*100:.2f}% |")
    lines+=['','## 클래스별 테스트 IoU','', '| 모델 | 차도(골목 포함) | 인도 | 자전거도로 |','|---|---:|---:|---:|']
    for key,name in [('A','B0'),('C','B2')]:
        vals=[next(r['iou'] for r in aggregates[key]['per_class'] if r['class_id']==cid)*100 for cid in ids]
        lines.append(f'| {name} | {vals[0]:.2f}% | {vals[1]:.2f}% | {vals[2]:.2f}% |')
    lines+=['','## 평가 범위와 한계','',
    '전체 픽셀 수를 합산한 confusion matrix로 클래스 IoU와 mIoU를 계산했다. ignore=255는 평가에서 제외한다. 검증 점수는 512×512, 테스트는 원본 크기로 복원한 예측으로 계산하므로 수치 조건도 구분한다.',
    '분할은 실제 장소명이 아닌 사용자 확인된 XML 촬영일 식별자와 dHash 유사 사진 기준이다. 실제 장소 독립성과 보행자 시점은 확인되지 않았다. 테스트는 1004 식별자 그룹이다.',
    '자전거도로 포함 사진은 학습19/검증3/테스트7장으로 적다. 단일 seed·소규모 데이터 결과이며 일반적인 국가별 성능이나 시각장애인 보행 안전을 보장하지 않는다.',
    '미표기·주의/공사·점자블록·횡단보도 속성은 제외됐고 기타·장애물 구분을 학습하지 않았다. 평가 결과는 주석된 3종 영역에 한정된다.',
    'Boundary F1은 원본 해상도에서 tolerance=2px이며 ignore 인접 경계는 제외한다. 추론 시간은 GPU forward와 원본 크기 복원 기준이며 파일 읽기·시각화·평가 시간은 포함하지 않는다.',
    '사용자가 이번 범위를 B0/B2 검수 라벨 실험으로 한정하여 B/D 자동 라벨 실험과 기존 5개 Teacher/기준 모델 비교는 실행하지 않았다.',
    '미리보기는 고정 테스트 목록의 처음 6장을 사용했고, 모든 pane에서 정답 ignore 영역은 어둡게 표시했다.','',
    '## 파일','', 'reports/B0_B2_test_summary.csv', 'reports/B0_B2_test_class_metrics.csv', 'reports/graphs/B0_B2_comparison.png', 'reports/B0_B2_test_examples.jpg',
    'checkpoints/A/best 및 checkpoints/C/best']
    report_path=reports/'B0_B2_comparison_report.md';report_path.write_text(chr(10).join(lines),encoding='utf8')
    result={'state':'completed','experiments':summary,'test_ids_sha256':hashlib.sha256(json.dumps(test_ids).encode()).hexdigest(),
            'parent_split_lock_sha256':digest(ROOT/'data/splits/locked.json'),'preview':str(preview),'report':str(report_path),
            'scope':'A and C only; no pseudo-label experiments or five-model baseline comparison',
            'test_results_used_to_retune':False,'class_metrics':per_class}
    (reports/'B0_B2_completion.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8')
    print('AC_COMPARISON_COMPLETE',json.dumps(result,ensure_ascii=False),flush=True)
    display(Image.open(preview))
    return result
