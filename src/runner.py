"""Every invocation is a fresh Python process. CLI failures are saved to logs."""
from pathlib import Path
import argparse
import json
import sys
import traceback
from .schema import *

def baseline_predictions(root,legacy_bundle,keys,split='test'):
    from .dataset import validate_frozen
    from .evaluation import evaluate_predictions
    import shutil
    sys.path.insert(0,str(Path(legacy_bundle).resolve()))
    import benchmark_core as legacy
    root=Path(root).resolve();data=validate_frozen(root);selected=[r for r in data if split=='all' or r['split']==split]
    if not selected or (split!='all' and any(r['label_verified']!='true' for r in selected)):raise ValueError('Reviewed test masks required')
    bundle=root/'baseline_benchmark';legacy.initialize(bundle)
    for row in selected:shutil.copyfile(root/row['image_path'],bundle/'inputs/other'/f'{row["image_id"]}.png')
    selected_ids={r['image_id'] for r in selected}
    manifest=legacy.inspect_inputs(bundle)
    manifest=manifest[manifest.image.map(lambda value:Path(value).stem in selected_ids)].copy()
    registry=load_json(Path(legacy_bundle)/'model_registry.json')
    verified=legacy.verify_models(bundle,registry,keys)
    for spec in verified.values():
        if spec.get('kind')=='original':spec['benchmark_root']=str(root)
    legacy.run_benchmark(bundle,manifest,verified,keys,require_gpu=True)
    for key in keys:
        support=[LEGACY_TO_TRAIN[c] for c in verified[key]['support'] if c in LEGACY_TO_TRAIN]
        dest=root/'outputs/masks'/key;dest.mkdir(parents=True,exist_ok=True);stats={}
        for path in (bundle/'outputs'/key).glob('*/*/record.json'):
            r=load_json(path)
            if r['status']!='success':continue
            image_id=Path(r['image']).stem
            if image_id not in selected_ids:continue
            mask=legacy_to_train(np.asarray(Image.open(path.parent/'common_ids.png')))
            Image.fromarray(mask).save(dest/f'{image_id}.png')
            stats[image_id]={'seconds':r['forward_postprocess_s'],'peak_gpu_mb':r.get('peak_allocated_mb')}
        if len(stats)!=len(selected):raise ValueError(f'{key}: incomplete baseline test predictions')
        save_json(dest/'run_info.json',stats);save_json(dest/'support.json',support)
        if split!='all':evaluate_predictions(root,key,dest,support,split,stats)

def report(root):
    from .dataset import refresh_reviews
    from .evaluation import plots
    root=Path(root);data=refresh_reviews(root);plots(root)
    histories=list((root/'checkpoints').glob('*/history.json'));evaluations=list((root/'metrics').glob('*/summary.csv'))
    text=['# 2차 실험 실행 기록','',f'생성 시각: {now()}',f'이미지: {len(data)}장',
          f'검수 완료 라벨: {sum(r["label_verified"]=="true" for r in data)}장',
          f'학습 기록: {len(histories)}개, 평가 요약: {len(evaluations)}개','',
          '## 실험 구성','A/C: 검수 라벨만, B/D: 동일 검수 라벨 + 별도 자동 라벨. 데이터량 효과와 라벨 품질 효과를 구분한다.',
          'B0/B2는 ImageNet 인코더에서 시작하며 분할 헤드는 새로 초기화한다. 과거 학습한 B0의 성능이 아니다.',
          '기본 9종에는 점자블록·계단이 포함되지 않는다. 보행 방향은 영상 좌표의 후보이며 실제 거리나 안전 경로가 아니다.','',
          '## 완료·미완료 구분']
    if not histories:text+=['실제 데이터 학습 미실행: 데이터/검수 라벨 확보 후 실행한다.']
    for p in histories:
        info=load_json(p.parent/'run.json');text += [f'- {p.parent.name}: {len(load_json(p))} epoch, 합성 검증 전용={info["tiny_synthetic_only"]}']
    text+=['','## 결과 해석','성능이 측정된 경우 metrics와 reports/graphs를 근거로 비교한다. 미지원 클래스는 N/A다.',
           '한국/비한국 보행자 그룹이 모두 확보되지 않으면 2:1 보조 점수는 계산하지 않는다.',
           '모델별 입력 해상도·정밀도·추론 구현 차이를 함께 보고한다.','',
           '## 수행평가 작성 항목','주제 / 기존 시스템 개선 동기 / 출처·이용 조건 / 데이터 검토 / 전처리 / X·y와 모델 설계 / 실제 성능 / 활용·한계 / 직접 작성한 느낀 점 / 전체 코드']
    (root/'reports/experiment_report.md').write_text('\n'.join(text),encoding='utf-8')

def main():
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['init','import','split','teachers','review-export','review-import','teacher-select','train','evaluate','student-features','baselines','benchmark','direction','report','smoke'])
    parser.add_argument('--root',required=True);parser.add_argument('--source');parser.add_argument('--legacy',default=str(Path(__file__).resolve().parents[1]/'legacy'))
    parser.add_argument('--experiment',default='A');parser.add_argument('--models',nargs='+',default=['mask2former_swin_l','oneformer_dinat_l','mask2former_r50','segformer_b5','segformer_b2'])
    parser.add_argument('--confirm-groups',action='store_true');parser.add_argument('--confirm-reviewed',action='store_true');parser.add_argument('--reviewer',default='')
    parser.add_argument('--config',default='{}');parser.add_argument('--split',default='test');args=parser.parse_args();root=init(args.root)
    import os
    os.environ['HF_HOME']=str(root/'checkpoints/huggingface_cache')
    os.environ['HF_HUB_CACHE']=str(root/'checkpoints/huggingface_cache/hub')
    try:
        if args.command=='init':return
        elif args.command=='import':
            from .dataset import import_images
            import_images(root,args.source)
        elif args.command=='split':
            from .dataset import split_dataset
            split_dataset(root,confirm_groups=args.confirm_groups)
        elif args.command=='teachers':
            from .labeling import generate_teacher_labels
            generate_teacher_labels(root,args.legacy)
        elif args.command=='review-export':
            from .labeling import export_review
            export_review(root,args.source or root/'data/review/review_bundle.zip')
        elif args.command=='review-import':
            from .labeling import import_review_batch
            import_review_batch(root,args.source,args.reviewer,args.confirm_reviewed)
        elif args.command=='teacher-select':
            from .labeling import select_teacher
            select_teacher(root)
        elif args.command=='train':
            from .training import train
            train(root,args.experiment,json.loads(args.config))
        elif args.command=='evaluate':
            from .training import predict_student
            predict_student(root,args.experiment,args.split)
        elif args.command=='student-features':
            from .training import predict_student
            for split in ['train','validation','test']:predict_student(root,args.experiment,split,evaluate=False)
        elif args.command=='baselines':baseline_predictions(root,args.legacy,args.models,args.split)
        elif args.command=='benchmark':
            sys.path.insert(0,str(Path(args.legacy).resolve()))
            import benchmark_core as legacy
            legacy.initialize(root);manifest=legacy.inspect_inputs(root)
            if manifest.empty:print('No exploration images; phase 1 results are preserved separately.')
            else:
                verified=legacy.verify_models(root,load_json(Path(args.legacy)/'model_registry.json'),args.models)
                legacy.run_benchmark(root,manifest,verified,args.models,require_gpu=True)
        elif args.command=='direction':
            from .direction import feature_table,classify
            for key in args.models:
                features=feature_table(root,key,root/'outputs/masks'/key)
                if not features.empty:classify(root,features,key)
        elif args.command=='smoke':
            from .smoke import smoke
            smoke(root)
        elif args.command=='report':report(root)
    except Exception as exc:
        save_json(root/'logs'/f'{args.command}_{now().replace(":","-")}.json',{'error':str(exc),'traceback':traceback.format_exc()})
        raise

if __name__=='__main__':main()
