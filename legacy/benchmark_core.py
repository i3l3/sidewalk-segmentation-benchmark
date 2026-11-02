"""보행자 시점 세그멘테이션 비교. benchmark.ipynb에도 같은 코드가 포함됩니다."""
from pathlib import Path
import contextlib
import gc
import hashlib
import importlib.metadata as metadata
import json
import platform
import random
import time
import traceback

import numpy as np
import pandas as pd
from PIL import Image, ImageOps, ImageDraw
from scipy.ndimage import binary_erosion, binary_dilation

IGNORE = 65535
CLASSES = {0: 'Other', 1: 'Road', 2: 'Sidewalk', 3: 'Curb', 4: 'Crosswalk (plain)',
           5: 'Tactile Paving', 6: 'Stairs', 7: 'Pedestrian Area', 8: 'Curb Cut',
           9: 'Crosswalk Marking', 10: 'Bike Lane', 11: 'Caution Zone'}
PALETTE = np.array([[65,65,65], [100,110,150], [225,95,170], [255,155,45],
                    [60,220,170], [255,225,20], [165,115,60], [150,90,225],
                    [255,185,130], [235,245,245], [50,160,255], [225,80,65]], dtype=np.uint8)
EXACT_MAP = {'road':1, 'roadway':1, 'sidewalk':2, 'curb':3, 'crosswalk - plain':4,
             'crosswalk':4, 'braille_guide_blocks':5, 'stairs':6, 'pedestrian area':7,
             'curb cut':8, 'lane marking - crosswalk':9, 'bike lane':10,
             'bike_lane':10, 'caution_zone':11}


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding='utf-8')


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()[:16]


def file_sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def environment():
    import torch
    packages = {}
    for name in ['torch', 'torchvision', 'transformers', 'huggingface-hub', 'numpy', 'pandas', 'Pillow', 'scipy']:
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            packages[name] = None
    return {'python': platform.python_version(), 'platform': platform.platform(), 'packages': packages,
            'cuda': torch.version.cuda, 'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            'vram_gb': torch.cuda.get_device_properties(0).total_memory / 2**30 if torch.cuda.is_available() else None}


def initialize(root, seed=42):
    import torch
    root = Path(root)
    for p in ['inputs/korea', 'inputs/other', 'ground_truth', 'notebooks', 'outputs', 'metrics', 'reports']:
        (root / p).mkdir(parents=True, exist_ok=True)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    dump(root / 'metrics/environment.json', environment())
    dump(root / 'ground_truth/class_schema.json', {'classes': CLASSES, 'ignore_id': IGNORE,
         'format': 'single-channel integer PNG, original image dimensions; unannotated=65535',
         'warning': 'Curb Cut, Pedestrian Area, Crosswalk Marking are separate classes. Caution Zone is not Stairs.'})
    return root


def mapping(id2label):
    labels = {int(k): str(v) for k, v in id2label.items()}
    table = {k: EXACT_MAP.get(v.strip().lower(), 0) for k,v in labels.items()}
    support = sorted(set(table.values()) | {0})
    rows = [{'native_id': k, 'native_name': v, 'common_id': table[k], 'common_name': CLASSES[table[k]]}
            for k,v in sorted(labels.items())]
    return table, support, rows


def remap(native, table):
    result = np.zeros(native.shape, dtype=np.uint16)
    unknown = set(np.unique(native)) - set(table)
    if unknown:
        raise ValueError(f'Unknown native IDs: {unknown}')
    for src, dst in table.items():
        result[native == src] = dst
    return result


def inspect_inputs(root):
    """이미지를 수집하지 않는다. 사용자 파일을 검사하고 CSV 명세서를 만든다."""
    root = Path(root)
    files = sorted(p for p in (root / 'inputs').rglob('*') if p.is_file() and p.suffix.lower() in {'.jpg','.jpeg','.png','.webp','.bmp'})
    rows = []
    for path in files:
        rel = path.relative_to(root / 'inputs').as_posix()
        row = {'image': rel, 'image_id': path.stem[:50] + '_' + hashlib.sha256(rel.encode()).hexdigest()[:10],
               'country': 'unknown', 'viewpoint': 'unknown', 'source_url': '', 'license': '',
               'location_group': '', 'split': 'test', 'gt_path': '', 'status': 'ok', 'error': ''}
        try:
            with Image.open(path) as im:
                row.update(original_mode=im.mode, exif_orientation=im.getexif().get(274, 1))
                rgb = ImageOps.exif_transpose(im).convert('RGB')
                row.update(width=rgb.width, height=rgb.height)
                # 같은 사진이 이름/확장자만 바뀌어도 중복을 확인한다.
                row['pixel_sha256'] = hashlib.sha256(rgb.tobytes() + str(rgb.size).encode()).hexdigest()
            row['file_sha256'] = file_sha(path)
        except Exception as exc:
            row.update(status='invalid_image', error=str(exc))
        rows.append(row)
    columns = ['image','image_id','country','viewpoint','source_url','license','location_group','split',
               'gt_path','status','error','original_mode','exif_orientation','width','height','pixel_sha256','file_sha256']
    frame = pd.DataFrame(rows, columns=columns)
    manifest = root / 'manifest.csv'
    if manifest.exists():
        old = pd.read_csv(manifest, keep_default_na=False).set_index('image')
        for index,row in frame.iterrows():
            if row['image'] in old.index:
                for key in ['country','viewpoint','source_url','license','location_group','split','gt_path']:
                    frame.at[index,key] = old.loc[row['image'],key] if key in old else ''
    frame['duplicate_of'] = ''
    seen = {}
    for idx,row in frame.iterrows():
        if row['status'] != 'ok':
            continue
        sha = row['pixel_sha256']
        if sha in seen:
            frame.at[idx,'duplicate_of'] = seen[sha]
        else:
            seen[sha] = row['image']
    frame.to_csv(manifest, index=False, encoding='utf-8-sig')
    frame.to_csv(root / 'metrics/data_inspection.csv', index=False, encoding='utf-8-sig')
    dump(root / 'metrics/data_summary.json', {'total':len(frame), 'invalid':int((frame.status!='ok').sum()),
         'duplicates':int((frame.duplicate_of!='').sum()), 'missing_metadata':{k:int((frame[k]=='').sum()) for k in ['source_url','license','location_group']},
         'note':'country/viewpoint must be supplied by the user; folder names do not prove provenance.'})
    return frame


def verify_models(root, registry, keys):
    """가중치 전체를 받기 전 API/설정/가중치 파일 존재 및 고정 리비전을 검증한다."""
    from huggingface_hub import HfApi, hf_hub_download
    verified = {}
    for key in keys:
        spec = dict(registry[key])
        try:
            if spec.get('kind')=='original':
                import ast
                import urllib.request
                from original_models import SPECS
                original=SPECS[spec['original_key']]
                owner=original['repo'].removeprefix('https://github.com/')
                base=f'https://raw.githubusercontent.com/{owner}/{original["revision"]}/'
                with urllib.request.urlopen(base+original['config']) as response:
                    config_text=response.read().decode()
                labels_path=original['package']+'/data/datasets/register_mapillary_vistas_panoptic.py'
                with urllib.request.urlopen(base+labels_path) as response:
                    tree=ast.parse(response.read().decode())
                categories=next(ast.literal_eval(node.value) for node in tree.body if isinstance(node,ast.Assign)
                    and any(isinstance(t,ast.Name) and t.id=='MAPILLARY_VISTAS_SEM_SEG_CATEGORIES' for t in node.targets))
                id2label={i:row['name'] for i,row in enumerate(categories)}
                table,support,rows=mapping(id2label)
                spec.update(status='metadata_verified',benchmark_root=str(Path(root).resolve()),
                            id2label=id2label,mapping=rows,support=support,
                            processor={'backend':'official_detectron2_config','config':original['config'],
                                       'config_text':config_text,'deform_attention':'official_pytorch_reference'},
                            weight_format='original -> safely decoded safetensors',source_spec=original)
                dump(Path(root)/'metrics/class_mappings'/f'{key}.json',spec)
                verified[key]=spec
                print(key,spec['status'],'(original isolated runtime)')
                continue
            info = HfApi().model_info(spec['model_id'], revision=spec['revision'])
            names = {s.rfilename for s in info.siblings}
            if not names.intersection({'model.safetensors','pytorch_model.bin','model.safetensors.index.json','pytorch_model.bin.index.json'}):
                raise ValueError('No recognized weight file')
            conf = json.loads(Path(hf_hub_download(spec['model_id'],'config.json',revision=info.sha)).read_text())
            proc = json.loads(Path(hf_hub_download(spec['model_id'],'preprocessor_config.json',revision=info.sha)).read_text())
            table,support,rows = mapping(conf['id2label'])
            spec.update(status='metadata_verified', revision=info.sha, id2label=conf['id2label'],
                        mapping=rows, support=support, processor=proc,
                        weight_format='safetensors' if names.intersection({'model.safetensors','model.safetensors.index.json'}) else 'pytorch_bin')
            dump(Path(root)/'metrics/class_mappings'/f'{key}.json', spec)
        except Exception as exc:
            spec.update(status='verification_failed', error=f'{type(exc).__name__}: {exc}')
        verified[key] = spec
        print(key, spec['status'])
    metadata_path=Path(root)/'metrics/model_metadata.json'
    existing=json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
    dump(metadata_path, {**existing,**verified})
    return verified


class HFAdapter:
    def __init__(self, spec, precision='auto', processor_size=None, require_gpu=True):
        import torch
        from transformers import (AutoImageProcessor, Mask2FormerForUniversalSegmentation,
                                  SegformerForSemanticSegmentation, OneFormerProcessor,
                                  OneFormerForUniversalSegmentation)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if require_gpu and self.device != 'cuda':
            raise RuntimeError('GPU가 없습니다. Colab 런타임 유형을 GPU로 바꾸세요. CPU 동작 확인은 REQUIRE_GPU=False.')
        self.kind = spec['kind']
        self.dtype = torch.float32
        if self.device == 'cuda' and precision != 'fp32':
            self.dtype = torch.bfloat16 if precision in {'auto','bf16'} and torch.cuda.is_bf16_supported() else torch.float16
        self.precision = str(self.dtype)
        kwargs = {'revision':spec['revision']}
        if self.kind == 'oneformer':
            self.processor = OneFormerProcessor.from_pretrained(spec['model_id'], **kwargs)
            cls = OneFormerForUniversalSegmentation
        else:
            self.processor = AutoImageProcessor.from_pretrained(spec['model_id'], use_fast=False, **kwargs)
            cls = Mask2FormerForUniversalSegmentation if self.kind == 'mask2former' else SegformerForSemanticSegmentation
        self.processor_size = processor_size
        # 크기 변경은 명시적으로 요청한 별도 실험에만 적용한다.
        if processor_size:
            target = self.processor.image_processor if self.kind == 'oneformer' else self.processor
            target.size = dict(processor_size)
        # .bin 전용 모델의 자동 변환 PR을 사용하지 않고 고정 리비전의 파일만 읽는다.
        self.model = cls.from_pretrained(spec['model_id'], use_safetensors=spec['weight_format']=='safetensors',
                                         **kwargs).to(self.device).eval()
        self.labels = self.model.config.id2label
        self.table,self.support,self.mapping_rows = mapping(self.labels)
        self.parameter_count = sum(p.numel() for p in self.model.parameters())

    def predict(self, image):
        import torch
        from torch.nn import functional as F
        args = {'images':image, 'return_tensors':'pt'}
        if self.kind == 'oneformer':
            args['task_inputs'] = ['semantic']
        inputs = self.processor(**args).to(self.device)
        input_shape = list(inputs['pixel_values'].shape)
        if self.device == 'cuda':
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        amp = torch.autocast('cuda', dtype=self.dtype) if self.device == 'cuda' and self.dtype != torch.float32 else contextlib.nullcontext()
        with torch.inference_mode(), amp:
            out = self.model(**inputs)
            if self.kind == 'segformer':
                pred = F.interpolate(out.logits.float(), size=(image.height,image.width), mode='bilinear', align_corners=False).argmax(1)[0]
            else:
                # 쿼리 마스크를 의미 클래스로 집계하는 공식 후처리. panoptic 인스턴스 ID를 쓰지 않는다.
                out.class_queries_logits = out.class_queries_logits.float()
                out.masks_queries_logits = out.masks_queries_logits.float()
                pred = self.processor.post_process_semantic_segmentation(out, target_sizes=[(image.height,image.width)])[0]
        if self.device == 'cuda':
            torch.cuda.synchronize()
        elapsed = time.perf_counter()-start
        peak = torch.cuda.max_memory_allocated()/2**20 if self.device == 'cuda' else None
        reserved = torch.cuda.max_memory_reserved()/2**20 if self.device == 'cuda' else None
        native = pred.cpu().numpy().astype(np.uint16)
        del inputs,out,pred
        return native, {'forward_postprocess_s':elapsed, 'peak_allocated_mb':peak,
                        'peak_reserved_mb':reserved, 'input_shape':input_shape, 'precision':self.precision,
                        'device':self.device, 'parameters':self.parameter_count}

    def close(self):
        import torch
        del self.model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def save_visuals(directory, image, native, common, labels):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    image.save(directory/'original.png')
    Image.fromarray(native.astype(np.uint16)).save(directory/'native_ids.png')
    Image.fromarray(common.astype(np.uint16)).save(directory/'common_ids.png')
    color = Image.fromarray(PALETTE[common])
    color.save(directory/'color.png')
    overlay = Image.blend(image, color, .5)
    overlay.save(directory/'overlay.png')
    # 저장용 미리보기만 축소한다. 평가 마스크는 원본 크기를 유지한다.
    thumb_size = (min(720,image.width), max(1,round(image.height*min(720/image.width,1))))
    panes = [im.resize(thumb_size,Image.Resampling.BILINEAR) for im in [image,color,overlay]]
    canvas = Image.new('RGB',(thumb_size[0]*3,thumb_size[1]+30),'white')
    draw = ImageDraw.Draw(canvas)
    for i,(pane,title) in enumerate(zip(panes,['Original','Common classes','Overlay (50%)'])):
        canvas.paste(pane,(i*thumb_size[0],30));draw.text((i*thumb_size[0]+8,8),title,fill='black')
    canvas.save(directory/'comparison.jpg',quality=93)
    dump(directory/'labels.json', {'native_id2label':labels,'common_id2label':CLASSES,'palette':PALETTE.tolist()})


def load_gt(root, row, shape):
    value = str(row.get('gt_path','')).strip()
    path = Path(root)/'ground_truth'/value if value else Path(root)/'ground_truth'/(row['image_id']+'.png')
    if not path.exists():
        return None
    # RGB 색상 마스크/임의 보간은 허용하지 않는다.
    a = np.asarray(Image.open(path))
    if a.ndim != 2 or a.dtype.kind not in 'ui':
        raise ValueError(f'GT must be integer class-ID PNG: {path}')
    if a.shape != shape:
        raise ValueError(f'GT/image size mismatch: {a.shape} vs {shape}; EXIF orientation must be aligned')
    bad = set(np.unique(a)) - (set(CLASSES)|{IGNORE})
    if bad:
        raise ValueError(f'Invalid GT IDs: {bad}. Use 65535 for ignore, not 255.')
    return a.astype(np.uint16)


def confusion(gt, pred, count=len(CLASSES)):
    valid = gt != IGNORE
    return np.bincount((gt[valid].astype(np.int64)*count+pred[valid]),minlength=count*count).reshape(count,count)


def metric_rows(cm, support):
    tp = np.diag(cm).astype(float)
    fp = cm.sum(0)-tp
    fn = cm.sum(1)-tp
    rows=[]
    for c,name in CLASSES.items():
        div = lambda a,b: float(a/b) if b else np.nan
        supported = c in support
        rows.append({'class_id':c,'class_name':name,'supported':supported,'gt_pixels':int(cm[c].sum()),
                     'pred_pixels':int(cm[:,c].sum()), 'iou':div(tp[c],tp[c]+fp[c]+fn[c]) if supported else np.nan,
                     'precision':div(tp[c],tp[c]+fp[c]) if supported else np.nan,
                     'recall':div(tp[c],tp[c]+fn[c]) if supported else np.nan,
                     'coverage_inclusive_iou':div(tp[c],tp[c]+fp[c]+fn[c])})
    # 공통 Road/Sidewalk의 FP에는 Other 영역의 오검출도 포함한다.
    common = [rows[c]['coverage_inclusive_iou'] for c in [1,2]]
    finite = lambda values: float(np.mean([v for v in values if np.isfinite(v)])) if any(np.isfinite(v) for v in values) else np.nan
    summary={'pixel_accuracy':float(tp.sum()/cm.sum()) if cm.sum() else np.nan,
             'shared_road_sidewalk_miou':finite(common),
             'coverage_inclusive_miou':finite([r['coverage_inclusive_iou'] for r in rows]),
             'supported_classes_miou':finite([r['iou'] for r in rows]),
             'valid_pixels':int(cm.sum()),'gt_supported_pixel_fraction':float(cm[support,:].sum()/cm.sum()) if cm.sum() else np.nan}
    return rows,summary


def boundary_counts(gt,pred,class_id,tolerance=2):
    valid=gt!=IGNORE
    safe=valid & ~binary_dilation(~valid,iterations=tolerance+1)
    boundary=lambda x: (x & ~binary_erosion(x,border_value=1)) & safe
    a=boundary(gt==class_id);b=boundary(pred==class_id)
    return np.array([int((b & binary_dilation(a,iterations=tolerance)).sum()),int(b.sum()),
                     int((a & binary_dilation(b,iterations=tolerance)).sum()),int(a.sum())],dtype=np.int64)


def boundary_f1(counts):
    matched_p,n_p,matched_g,n_g=counts
    if not n_p and not n_g:
        return np.nan
    p=matched_p/n_p if n_p else 0.
    r=matched_g/n_g if n_g else 0.
    return float(2*p*r/(p+r)) if p+r else 0.


def evaluate_saved(root, manifest, selected_keys=None, boundary_tolerance=2):
    root=Path(root);records=[];summaries=[];failures=[]
    lookup={r['image_id']:r for r in manifest.to_dict('records')}
    aggregates={}
    for record_path in sorted((root/'outputs').glob('*/*/*/record.json')):
        record=json.loads(record_path.read_text())
        if record.get('status')!='success' or (selected_keys and record['model'] not in selected_keys):
            continue
        row=lookup.get(record['image_id'])
        if row is None or row['status']!='ok' or row['duplicate_of']:
            continue
        if record.get('file_sha256') != row['file_sha256']:
            failures.append({'image_id':row['image_id'],'record':str(record_path),'error':'Input changed since prediction; rerun inference.'})
            continue
        try:
            pred=np.asarray(Image.open(record_path.parent/'common_ids.png'))
            gt=load_gt(root,row,pred.shape)
            if gt is None:
                continue
            cm=confusion(gt,pred)
            if not cm.sum():
                continue
            support=record['support']
            scores,summary=metric_rows(cm,support)
            for score in scores:
                c=score['class_id']
                counts=boundary_counts(gt,pred,c,boundary_tolerance)
                score['boundary_f1']=boundary_f1(counts) if c in support else np.nan
                records.append({**{k:record[k] for k in ['model','condition','image_id']},**score})
            groups=[('all', 'all')]
            if str(row['country']).lower() in {'korea','kr','south korea','한국'}:
                groups.append(('country','korea'))
            if str(row['viewpoint']).lower()=='pedestrian':
                groups.append(('viewpoint','pedestrian'))
            for group,value in groups:
                key=(record['model'],record['condition'],group,value)
                if key not in aggregates:
                    aggregates[key]={'cm':np.zeros_like(cm),'boundaries':np.zeros((len(CLASSES),4),dtype=np.int64),'support':support,'n':0}
                a=aggregates[key];a['cm']+=cm;a['n']+=1
                for c in CLASSES:
                    a['boundaries'][c]+=boundary_counts(gt,pred,c,boundary_tolerance)
        except Exception as exc:
            failures.append({'image_id':row['image_id'],'record':str(record_path),'error':str(exc)})
    aggregate_classes=[]
    for (model,condition,group,value),a in aggregates.items():
        rows,summary=metric_rows(a['cm'],a['support'])
        info={'model':model,'condition':condition,'group':group,'group_value':value,'n_images':a['n']}
        summaries.append({**info,**summary,'boundary_tolerance_px':boundary_tolerance})
        for r in rows:
            r['boundary_f1']=boundary_f1(a['boundaries'][r['class_id']]) if r['supported'] else np.nan
            aggregate_classes.append({**info,**r})
    pd.DataFrame(records).to_csv(root/'metrics/accuracy_per_image.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(aggregate_classes).to_csv(root/'metrics/accuracy_results.csv',index=False,encoding='utf-8-sig')
    result=pd.DataFrame(summaries)
    result.to_csv(root/'metrics/accuracy_summary.csv',index=False,encoding='utf-8-sig')
    dump(root/'metrics/ground_truth_errors.json',failures)
    return result


def collect_inference_results(root):
    root=Path(root);rows=[]
    for p in sorted((root/'outputs').glob('*/*/*/record.json')):
        r=json.loads(p.read_text())
        flat={k:v for k,v in r.items() if not isinstance(v,(dict,list))}
        flat['input_shape']=json.dumps(r.get('input_shape'))
        rows.append(flat)
    frame=pd.DataFrame(rows)
    frame.to_csv(root/'metrics/inference_results.csv',index=False,encoding='utf-8-sig')
    if not frame.empty and 'forward_postprocess_s' in frame:
        ok=frame[frame.status=='success'].copy()
        summary=ok.groupby(['model','condition','device','input_shape'],dropna=False).agg(
            n_images=('image_id','count'), mean_seconds=('forward_postprocess_s','mean'),
            median_seconds=('forward_postprocess_s','median'),std_seconds=('forward_postprocess_s','std'),
            max_peak_allocated_mb=('peak_allocated_mb','max'))
        summary.to_csv(root/'metrics/speed_summary.csv',encoding='utf-8-sig')
    return frame


def run_benchmark(root, manifest, verified, keys, require_gpu=True, precision='auto',
                  processor_overrides=None, limit=None, resume=True, warmup=True):
    root=Path(root);processor_overrides=processor_overrides or {}
    images=manifest[(manifest.status=='ok') & (manifest.duplicate_of=='')]
    if limit:
        images=images.head(limit)
    if images.empty:
        print('입력 이미지가 없습니다. inputs에 사진을 추가한 뒤 데이터 점검 셀부터 다시 실행하세요.')
        return pd.DataFrame()
    import torch
    if require_gpu and not torch.cuda.is_available():
        raise RuntimeError('GPU 런타임이 필요합니다. 런타임 > 런타임 유형 변경 > GPU')
    env=environment()
    active_conditions={}
    for key in keys:
        adapter=None
        spec=verified[key]
        override=processor_overrides.get(key)
        condition=fingerprint({'revision':spec.get('revision'),'processor':spec.get('processor'),
                               'override':override,'precision':precision,'environment':env,'pipeline_version':1})
        active_conditions[key]=condition
        model_dir=root/'outputs'/key/condition
        model_dir.mkdir(parents=True,exist_ok=True)
        dump(model_dir/'condition.json',{'model':key,'spec':spec,'override':override,'environment':env,
             'condition':condition,'default_processor':not bool(override),
             'time_definition':'forward + official semantic postprocessing; excludes download, loading, CPU preprocessing and file I/O. End-to-end is separate.'})
        try:
            if spec.get('status')!='metadata_verified':
                raise RuntimeError(spec.get('error','metadata not verified'))
            pending=[]
            for row in images.to_dict('records'):
                output=model_dir/row['image_id']
                previous=output/'record.json'
                if resume and previous.exists():
                    old=json.loads(previous.read_text())
                    if old.get('status')=='success' and old.get('file_sha256')==row['file_sha256'] and all((output/p).exists() for p in ['native_ids.png','common_ids.png','overlay.png','comparison.jpg']):
                        continue
                pending.append(row)
            if not pending:
                print(key,'saved results reused');continue
            start=time.perf_counter()
            if spec.get('kind')=='original':
                from original_models import OriginalAdapter
                adapter=OriginalAdapter(spec,precision=precision,processor_size=override,require_gpu=require_gpu)
            else:
                adapter=HFAdapter(spec,precision=precision,processor_size=override,require_gpu=require_gpu)
            load_s=time.perf_counter()-start
            first=True
            for row in pending:
                output=model_dir/row['image_id']
                record={'model':key,'condition':condition,'image_id':row['image_id'], 'image':row['image'],
                        'file_sha256':row['file_sha256'],'revision':spec['revision'],'support':adapter.support,
                        'load_seconds':load_s,'country':row['country'],'viewpoint':row['viewpoint']}
                try:
                    with Image.open(root/'inputs'/row['image']) as opened:
                        image=ImageOps.exif_transpose(opened).convert('RGB')
                    if first and warmup:
                        adapter.predict(image)
                    first=False
                    begin=time.perf_counter()
                    native,stats=adapter.predict(image)
                    e2e=time.perf_counter()-begin
                    common=remap(native,adapter.table)
                    save_visuals(output,image,native,common,adapter.labels)
                    record.update(status='success',end_to_end_s=e2e,**stats,
                                  width=image.width,height=image.height,
                                  common_pixel_ratios={CLASSES[c]:float((common==c).mean()) for c in CLASSES},
                                  native_pixel_ratios={str(c):float((native==c).mean()) for c in adapter.labels})
                    dump(output/'record.json',record)
                    print(key,row['image'],'OK',round(stats['forward_postprocess_s'],3),'s')
                except Exception as exc:
                    record.update(status='oom' if isinstance(exc,torch.cuda.OutOfMemoryError) else 'failed',
                                  error=f'{type(exc).__name__}: {exc}', traceback=traceback.format_exc())
                    dump(output/'record.json',record)
                    print(key,row['image'],record['status'],str(exc))
                    if isinstance(exc,torch.cuda.OutOfMemoryError):
                        gc.collect();torch.cuda.empty_cache()
                        print('크기를 자동으로 바꾸지 않았습니다. processor_overrides에 축소 크기를 지정해 별도 조건으로 다시 실행하세요.')
                        break
        except Exception as exc:
            dump(model_dir/'model_failure.json',{'model':key,'status':'load_failed','error':f'{type(exc).__name__}: {exc}', 'traceback':traceback.format_exc()})
            print(key,'load failed:',exc)
        finally:
            if adapter is not None:
                adapter.close()
            adapter=None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    frame=collect_inference_results(root)
    if frame.empty:
        return frame
    return frame[[active_conditions.get(r['model'])==r['condition'] for r in frame.to_dict('records')]].copy()


def comparison_grids(root, manifest):
    root=Path(root);templates=[];key=[]
    for row in manifest.to_dict('records'):
        paths=sorted((root/'outputs').glob(f'*/*/{row["image_id"]}/overlay.png'))
        panels=[]
        for idx,path in enumerate(paths):
            record=json.loads((path.parent/'record.json').read_text())
            if record.get('status')!='success':
                continue
            if record.get('file_sha256') != row['file_sha256']:
                continue
            blind='M'+fingerprint({'model':record['model'],'condition':record['condition']})[:6]
            panel=Image.open(path).convert('RGB');panel.thumbnail((600,420))
            card=Image.new('RGB',(600,460),'white');card.paste(panel,(0,35))
            ImageDraw.Draw(card).text((10,10),blind,fill='black');panels.append(card)
            key.append({'image_id':row['image_id'],'blind_id':blind,'model':record['model'],'condition':record['condition']})
            templates.append({'image_id':row['image_id'],'blind_id':blind,'pedestrian_score':'','korea_score':'',
                              'pedestrian_reason':'','korea_reason':'','rater':'','unsupported_limitations':''})
        if panels:
            grid=Image.new('RGB',(600*min(3,len(panels)),460*((len(panels)+2)//3)),'white')
            for i,panel in enumerate(panels):
                grid.paste(panel,((i%3)*600,(i//3)*460))
            target=root/'reports/blind_comparisons'/f'{row["image_id"]}.jpg'
            target.parent.mkdir(parents=True,exist_ok=True);grid.save(target,quality=93)
    pd.DataFrame(key).to_csv(root/'reports/blind_key.csv',index=False,encoding='utf-8-sig')
    template=root/'reports/qualitative_scores.csv'
    if templates:
        new=pd.DataFrame(templates)
        if template.exists():
            old=pd.read_csv(template,keep_default_na=False)
            existing=set(zip(old.image_id,old.blind_id))
            new=new[[pair not in existing for pair in zip(new.image_id,new.blind_id)]]
            new=pd.concat([old,new],ignore_index=True)
        new.to_csv(template,index=False,encoding='utf-8-sig')
    return pd.DataFrame(key)


def qualitative_summary(root, manifest):
    root=Path(root);path=root/'reports/qualitative_scores.csv';key=root/'reports/blind_key.csv'
    if not path.exists() or not key.exists():
        return pd.DataFrame()
    scores=pd.read_csv(path);keys=pd.read_csv(key)
    merged=scores.merge(keys,on=['image_id','blind_id'],validate='many_to_one').merge(manifest[['image_id','country','viewpoint']],on='image_id',validate='many_to_one')
    for col in ['pedestrian_score','korea_score']:
        merged[col]=pd.to_numeric(merged[col],errors='coerce')
        if (merged[col].notna() & ~merged[col].between(1,5)).any():
            raise ValueError('정성 평가는 1~5점 또는 공란이어야 합니다.')
    merged.loc[merged.viewpoint!='pedestrian','pedestrian_score']=np.nan
    merged.loc[~merged.country.str.lower().isin(['korea','kr','south korea','한국']),'korea_score']=np.nan
    # 평가자가 많은 이미지에 가중치가 몰리지 않도록 이미지별 평균 후 집계한다.
    per_image=merged.groupby(['model','condition','image_id'])[['pedestrian_score','korea_score']].mean()
    result=per_image.groupby(['model','condition']).mean()
    counts=per_image.groupby(['model','condition']).count().rename(columns={'pedestrian_score':'n_pedestrian_images','korea_score':'n_korea_images'})
    result=result.join(counts)
    result['weighted_score']=(result.pedestrian_score*2+result.korea_score)/3
    result=result.reset_index()
    result.to_csv(root/'metrics/qualitative_summary.csv',index=False,encoding='utf-8-sig')
    return result


def make_report(root, manifest, verified, accuracy, qualitative):
    root=Path(root)
    collect_inference_results(root)
    lines=['# 보행자 시점 인도 세그멘테이션 비교 보고서', '',
           '## 1. 주제와 탐구 질문',
           '공개 사전학습 모델은 한국 보행자 시점에서 인도·차도 및 세부 구조를 얼마나 잘 구분하는가?',
           'X는 RGB 이미지의 픽셀 값(H×W×3), y는 각 픽셀의 의미 클래스다. 정답은 사람이 검수한 마스크이며 예측 마스크가 아니다.',
           '모델 종류·학습 데이터·입력 해상도는 실험 조건이다. 이 실험은 전이된 사전학습 모델의 추론 비교이며 새 학습을 수행하지 않는다.', '',
           '## 2. 데이터 수집 및 검토',f'등록 이미지 {len(manifest)}장. 출처·이용 조건·촬영 국가·시점은 manifest.csv에서 확인한다.',
           '데이터 수집은 아직 별도 과제다. 사진 4장으로 일반적인 한국 환경 성능을 확정할 수 없다. 같은 장소 연속 사진은 location_group으로 묶는다.',
           'data_inspection.csv에는 파일 손상, 크기, 모드, 중복, 결측 메타데이터가 기록된다. 변수 목록과 자료형은 manifest.dtypes/describe 출력으로 확인한다.', '',
           '## 3. 전처리',
           '손상 파일과 중복 이미지를 제외한다. 사진의 EXIF 방향을 반영하고 RGB로 변환한다. 극단적인 해상도는 자동 제거하지 않고 점검한다.',
           '정규화와 입력 크기는 각 모델의 고정 리비전 공식 Processor를 사용한다. 정수 마스크는 무손실 PNG로 보존한다.',
           '정답 누락은 65535(ignore)로 제외한다. RGB 정답이나 크기가 맞지 않는 정답을 임의 변환하지 않는다.', '',
           '## 4. 모델 설계와 지원 범위']
    for key,spec in verified.items():
        names=', '.join(CLASSES[c] for c in spec.get('support',[]) if c)
        lines += [f'- **{key}**: {spec["model_id"]} / `{spec.get("revision")}` / {spec.get("status")}',
                  f'  - 지원 클래스: {names or "검증 실패"}. 라이선스: {spec.get("license_note","확인 필요")}.',
                  f'  - 출처: {spec.get("source_spec",{}).get("repo", "https://huggingface.co/"+spec["model_id"])}']
    lines += ['', '점자블록은 Mapillary/Cityscapes 후보에서 독립 지원되지 않는다. Curb Cut은 Curb와, Crosswalk Marking은 Crosswalk (plain)과 합치지 않았다.',
              'PSPUNet의 Caution Zone은 계단 외 영역도 포함하므로 Stairs로 바꾸지 않는다.', '',
              '## 5. 모델 성능과 분석',
              '추론 시간·메모리: metrics/inference_results.csv. 실패와 OOM은 outputs의 model_failure.json/record.json에 원인과 함께 기록된다.',
              '시간은 다운로드·모델 로딩·파일 저장을 제외한 forward+후처리이며 전처리 포함 시간은 end_to_end_s다. 워밍업 1회 뒤 이미지당 1회 측정이므로 정밀 속도 벤치마크는 아니다.',
              'GPU 메모리는 PyTorch peak allocated/reserved 값이며 전체 장치 사용량과 다르다. CPU 실행은 GPU 측정값이 비어 있다.',
              '각기 다른 기본 Processor 해상도 때문에 속도·경계 차이를 백본 크기만의 효과로 해석하면 안 된다.', '',
              '정답이 없으면 픽셀 비율은 출력 분포일 뿐 정확도가 아니다. accuracy_summary.csv는 검수 GT가 있을 때만 작성된다.',
              'shared_road_sidewalk_miou는 모든 모델의 공통 Road/Sidewalk IoU 평균이다. coverage_inclusive_miou는 미지원 GT 클래스 오류도 포함한다.',
              'supported_classes_miou는 지원 범위가 다르면 직접 순위를 비교하지 않는다. 개별 미지원 클래스는 N/A(빈칸)다. GT에도 예측에도 없는 클래스는 평균에서 제외한다.',
              'Boundary F1은 원본 해상도에서 2px 허용 오차를 쓴다. ignore 인접 경계는 제외한다. 이미지 크기에 따라 2px의 상대적 의미가 다르므로 함께 검토한다.']
    if not accuracy.empty:
        lines += ['', '```text', accuracy.round(4).to_string(index=False), '```']
    else:
        lines += ['', '**정량 정확도 결과 없음: 검수된 정답 마스크가 아직 없거나 유효한 예측/정답 쌍이 없다.**']
    if (root/'metrics/speed_summary.csv').exists():
        speed=pd.read_csv(root/'metrics/speed_summary.csv')
        lines += ['', '### 추론 시간·PyTorch GPU 메모리', '```text',speed.round(3).to_string(index=False),'```']
    lines += ['', '## 6. 정성 평가와 최종 선택',
              'reports/blind_comparisons의 동일 이미지 비교를 보고 qualitative_scores.csv에 1~5점과 판단 근거를 기입한다.',
              '보행자 시점: 인도·차도 경계, 얇은 구조 연속성, 배경 침범. 한국 환경: 국내 보도블록·횡단보도·골목의 안정성.',
              '1점=광범위한 누락/오분류, 2점=여러 핵심 구역 오류, 3점=넓은 영역 양호·세부 오류, 4점=대체로 정확·일부 오류, 5점=검토한 필수 요소에 거의 오류 없음.',
              '관측되지 않은 구조나 미지원 클래스가 있으면 반드시 한계란에 적는다. 시점·국가가 미확인인 이미지에는 해당 점수를 집계하지 않는다.',
              '종합점수=(보행자 시점 점수×2+한국 환경 점수)/3. 정성 점수이며 논문 mIoU가 아니다. 두 점수가 모두 있을 때만 산출한다.']
    if not qualitative.empty:
        lines += ['', '```text', qualitative.round(3).to_string(index=False), '```',
                  '동일 이미지/조건의 평가 수와 클래스 지원 범위를 함께 검토한 뒤 최종 모델을 선택한다. 자동으로 우승을 확정하지 않는다.']
    else:
        lines += ['', '**최종 추천 보류: 실제 한국 보행 이미지 및 평가 점수가 준비되지 않았다.**']
    lines += ['', '## 7. 예상과 실제 결과 / 개선 방안',
              '가설: Mapillary 후보가 Cityscapes 후보보다 연석·횡단보도 세부 구조에 유리할 수 있다. 실제 결과를 입력한 뒤 확인한다.',
              '차이를 모두 학습 데이터 효과로 돌릴 수 없다. 아키텍처·규모·해상도·학습 방식도 함께 다르다.',
              '경계 오차가 남으면 검수 라벨 품질, 입력 해상도, 얇은 클래스 표본 수를 먼저 점검한다. 추가 학습은 별도 결정한다.',
              'PSPUNet 비교는 동일 입력 이미지의 결과를 확보한 후에만 개선 여부를 판단한다.', '',
              '## 8. 결론 및 활용 / 느낀 점',
              '실제 평가 후 가장 적합한 모델, 한국 환경 안정성, 인도·차도 경계, 세부 구조, PSPUNet 대비 개선, 추가 학습 필요성을 근거와 함께 작성한다.',
              '보행 보조 연구용 비교이며 실제 안전 보행을 보장하지 않는다. 느낀 점은 본인이 겪은 문제·해결 과정과 함께 직접 작성한다.', '',
              '## 9. 재현 및 전체 코드',
              'notebooks/benchmark.ipynb와 benchmark_core.py, model_registry.json 및 environment.json을 함께 보관한다.',
              '합성 이미지 smoke test는 소프트웨어 동작 확인이며 실제 모델 성능 결과에 포함하지 않는다.']
    path=root/'reports/comparison_report.md';path.write_text('\n'.join(lines),encoding='utf-8')
    return path
