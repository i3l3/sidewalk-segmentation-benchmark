"""Image-coordinate walking-direction candidates, using the trained three-class B2.

This is a rule baseline, not a safety/physical-distance estimator or direction ground truth.
"""
from pathlib import Path
from collections import Counter
import heapq
import json
import math
from time import perf_counter
import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import distance_transform_edt
from .schema import read_mask, load_json, save_json, sha, decode_task_mask, active_class_ids, COLORS, now
from .dataset import validate_frozen
from .direction import extract_features, rule_direction, DEFAULT_RULE

ROUTE_DEFAULTS = {**DEFAULT_RULE, 'grid_width': 192, 'lookahead_y': .55, 'clearance_weight': 2.0, 'straight_half_width': .1}
LABELS = ['Left', 'Straight', 'Right', 'No clear path']


def _line_pixels(a, b):
    steps = max(abs(b[0] - a[0]), abs(b[1] - a[1])) + 1
    yy = np.rint(np.linspace(a[0], b[0], steps)).astype(int)
    xx = np.rint(np.linspace(a[1], b[1], steps)).astype(int)
    return yy, xx


def direction_candidate(mask, obstacle_mask=None, config=None):
    config = {**ROUTE_DEFAULTS, **(config or {})}
    mask = np.asarray(mask)
    if mask.ndim != 2 or min(mask.shape) < 8:
        raise ValueError('A 2D segmentation mask of at least 8x8 is required')
    if not 0 < config['lookahead_y'] < .9 or not 16 <= config['grid_width'] <= 512 or not 0 < config['straight_half_width'] < .4:
        raise ValueError('Invalid route grid or lookahead')
    features, connected = extract_features(mask, obstacle_mask)
    # The B2 task does not contain a curb or obstacle class: zero would imply a measurement.
    features['curb_proximity'] = None
    result = {'direction': 'No clear path', 'reason': None, 'features': features,
              'scores': {name: None for name in LABELS[:3]}, 'path_xy_normalized': [],
              'lookahead_y_normalized': config['lookahead_y'],
              'curb_feature_status': 'unsupported_by_three_class_model',
              'obstacle_feature_status': 'external_mask' if obstacle_mask is not None else 'not_available',
              'units': 'normalized image coordinates; no meters'}
    if rule_direction(features, config) == 'No clear path':
        result['reason'] = 'no_bottom_connected_region_or_insufficient_relative_width_continuity'
        return result, connected
    h, w = mask.shape
    gw = min(int(config['grid_width']), w)
    gh = max(8, round(h * gw / w))
    grid = np.asarray(Image.fromarray(connected.astype(np.uint8)).resize((gw, gh), Image.Resampling.NEAREST)).astype(bool)
    # Four-neighbor routes cannot jump diagonally across a road corner.
    seeds = np.argwhere(grid & (np.indices(grid.shape)[0] >= int(.95 * gh)) &
                        (np.indices(grid.shape)[1] >= int(.4 * gw)) &
                        (np.indices(grid.shape)[1] < max(int(.4 * gw)+1, int(.6 * gw))))
    if not len(seeds):
        result['reason'] = 'no_seed_on_route_grid'
        return result, connected
    start = tuple(min(seeds.tolist(), key=lambda p: (-p[0], abs(p[1] - .5 * (gw-1)))))
    target_y = int(round(config['lookahead_y'] * (gh-1)))
    if target_y >= start[0]:
        result['reason'] = 'insufficient_forward_extent'
        return result, connected
    clearance = distance_transform_edt(np.pad(grid, 1, constant_values=False))[1:-1, 1:-1]
    distance = np.full(grid.shape, np.inf)
    distance[start] = 0
    parents = {}
    queue = [(0., start)]
    while queue:
        cost, (y, x) = heapq.heappop(queue)
        if cost != distance[y, x]:
            continue
        # Moving down would turn the on-screen candidate into a backwards detour.
        for ny, nx in ((y-1, x), (y, x-1), (y, x+1)):
            if ny < target_y or not (0 <= nx < gw) or not grid[ny, nx]:
                continue
            candidate = cost + 1 + config['clearance_weight'] / (clearance[ny, nx] + 1)
            if candidate < distance[ny, nx]:
                distance[ny, nx] = candidate
                parents[(ny, nx)] = (y, x)
                heapq.heappush(queue, (candidate, (ny, nx)))
    paths = {}
    for band, name in enumerate(LABELS[:3]):
        def sector(x):
            value=x/max(gw-1,1)
            return 0 if value<.5-config['straight_half_width'] else 2 if value>.5+config['straight_half_width'] else 1
        xs = [x for x in range(gw) if sector(x) == band and np.isfinite(distance[target_y, x])]
        if not xs:
            continue
        x = max(xs, key=lambda col: clearance[target_y, col]/gw - .05*distance[target_y, col]/gh)
        path = [(target_y, x)]
        while path[-1] != start:
            path.append(parents[path[-1]])
        path.reverse()
        source_path = [(min(h-1, int((y+.5)*h/gh)), min(w-1, int((x+.5)*w/gw))) for y, x in path]
        # Nearest resize may bridge a sub-grid hole. Reject it at source resolution.
        if not all(connected[_line_pixels(a, b)].all() for a, b in zip(source_path, source_path[1:])):
            continue
        width = 2 * float(np.percentile([clearance[p] for p in path], 25)) / gw
        score = max(0., min(1., width) - .02*len(path)/gh)
        result['scores'][name] = score
        paths[name] = source_path
    available = {k: v for k, v in result['scores'].items() if v is not None}
    if not available:
        result['reason'] = 'no_verified_connected_route_to_lookahead'
        return result, connected
    chosen = max(available, key=available.get)
    if 'Straight' in available and available['Straight'] >= available[chosen] - config['center_margin']:
        chosen = 'Straight'
    result.update(direction=chosen, reason='connected_relative_width_route_baseline',
                  path_xy_normalized=[[float(x/max(w-1,1)), float(y/max(h-1,1))] for y, x in paths[chosen]])
    return result, connected


def render_candidate(image, mask, result, connected):
    image = image.convert('RGB')
    pixels = np.asarray(image).copy()
    pixels[connected] = (pixels[connected]*.6 + np.array([0, 210, 130])*.4).astype(np.uint8)
    canvas = Image.fromarray(pixels)
    draw = ImageDraw.Draw(canvas); w, h = canvas.size
    route = [(round(x*(w-1)), round(y*(h-1))) for x, y in result['path_xy_normalized']]
    line_width = max(3, w//150)
    if route:
        draw.line(route, fill='yellow', width=line_width)
        end = route[-1]; earlier = route[max(0, len(route)-7)]
        angle = math.atan2(end[1]-earlier[1], end[0]-earlier[0]); size = max(12, w//40)
        head = [end, (end[0]-size*math.cos(angle-.5), end[1]-size*math.sin(angle-.5)),
                (end[0]-size*math.cos(angle+.5), end[1]-size*math.sin(angle+.5))]
        draw.polygon(head, fill='yellow')
        start = route[0]; draw.ellipse((start[0]-6,start[1]-6,start[0]+6,start[1]+6),fill='white')
    else:
        x, y = w//2, int(.85*h); size = max(12, w//35)
        draw.line((x-size,y-size,x+size,y+size), fill='red', width=line_width)
        draw.line((x-size,y+size,x+size,y-size), fill='red', width=line_width)
    canvas.thumbnail((960, 540))
    # Text at preview scale remains readable, unlike text rasterized before downsampling.
    output = Image.new('RGB', (canvas.width, canvas.height+58), 'white'); output.paste(canvas, (0,58))
    draw = ImageDraw.Draw(output)
    draw.text((8,7), 'B2 direction candidate: '+result['direction'], fill='black')
    draw.text((8,25), 'Green: bottom-connected sidewalk | Yellow: verified image-coordinate route', fill='black')
    draw.text((8,42), 'Rule baseline; obstacle/curb detection unavailable. Not a safety guarantee.', fill='black')
    return output


def _checkpoint(root):
    root = Path(root)
    run = load_json(root/'checkpoints/C/run.json')
    if run.get('tiny_synthetic_only') or run['config'].get('class_ids') != [1,2,8] or active_class_ids(root) != [1,2,8]:
        raise ValueError('Real, trained three-class B2 checkpoint is required')
    weight = root/'checkpoints/C/best/model.safetensors'
    digest = sha(weight)
    comparison = root/'reports/B0_B2_completion.json'
    if comparison.exists():
        expected = next(r['best_weights_sha256'] for r in load_json(comparison)['experiments'] if r['experiment']=='C')
        if digest != expected:raise ValueError('Best B2 checkpoint changed since fixed-test evaluation')
    return run, digest


def run_b2_direction(root, split='test'):
    root = Path(root); run, checkpoint_sha = _checkpoint(root)
    data = validate_frozen(root)
    selected = [row for row in data if row['split']==split]
    if not selected:raise ValueError('No selected images')
    rule_path = root/'data/splits/b2_route_rule_v1.json'
    declaration = {'version':1, 'config':ROUTE_DEFAULTS, 'reason':'predeclared_defaults_no_direction_validation_labels',
                   'direction_labels_used_for_tuning':[], 'model':'C / SegFormer B2', 'checkpoint_sha256':checkpoint_sha}
    if rule_path.exists():
        existing = load_json(rule_path)
        if any(existing[k]!=declaration[k] for k in declaration):raise ValueError('Frozen route rule changed; create a new explicit rule version')
    else:save_json(rule_path,{**declaration,'created_at':now()})
    out = root/'outputs/b2_direction';out.mkdir(exist_ok=True,parents=True)
    records = []
    for i,row in enumerate(selected):
        mask = read_mask(root/'outputs/masks/C'/(row['image_id']+'.png'),(int(row['image_height']),int(row['image_width'])))
        if set(np.unique(mask)) - {1,2,8,255}:raise ValueError('Prediction is not a canonical three-class B2 mask')
        started = perf_counter()
        result, connected = direction_candidate(mask)
        result['route_seconds'] = perf_counter() - started
        result.update(image_id=row['image_id'], split=split, source_model='SegFormer B2 C/best', checkpoint_sha256=checkpoint_sha)
        save_json(out/(row['image_id']+'.json'), result)
        Image.fromarray((connected*255).astype(np.uint8)).save(out/(row['image_id']+'_connected.png'))
        with Image.open(root/row['image_path']) as image:
            render_candidate(image,mask,result,connected).save(out/(row['image_id']+'_direction.jpg'),quality=92)
        records.append({'image_id':row['image_id'],'split':split,'group_id':row['group_id'],
                        'direction_candidate':result['direction'],'reason':result['reason'], 'route_seconds':result['route_seconds'], **result['features'],
                        **{k+'_route_score':v for k,v in result['scores'].items()},
                        'path_xy_normalized':json.dumps(result['path_xy_normalized']),
                        'human_direction_label':row['direction_label'] if row['direction_verified']=='true' else '',
                        'human_direction_verified':row['direction_verified']})
        if (i+1)%10==0:print(f'B2 direction candidates: {i+1}/{len(selected)}',flush=True)
    import pandas as pd
    frame = pd.DataFrame(records);frame.to_csv(root/'metrics/B2_direction_candidates.csv',index=False)
    verified = [r for r in selected if r['direction_verified']=='true']
    summary = {'model':'B2 C/best','checkpoint_sha256':checkpoint_sha,'images':len(records),'split':split,
               'candidate_counts':dict(Counter(r['direction_candidate'] for r in records)),
               'route_latency_seconds':{k:float(v) for k,v in zip(['mean','median','p95'],[frame.route_seconds.mean(),frame.route_seconds.median(),frame.route_seconds.quantile(.95)])},
               'route_latency_scope':'cached full-resolution mask to candidate; excludes file IO and rendering',
               'no_clear_path_fraction':sum(r['direction_candidate']=='No clear path' for r in records)/len(records),
               'verified_human_direction_labels':len(verified),'direction_accuracy':None,
               'direction_accuracy_reason':'No direction evaluation is reported; human direction labels and a separate protocol are required',
               'decision_tree_random_forest_trained':False,'classifier_reason':'Need >=20 train /5 validation /5 test human labels per direction',
               'thresholds_tuned_on_test':False,'curb_and_obstacle_detection':'not_available_in_three_class_model',
               'rule_path':str(rule_path),'output_directory':str(out),'units':'normalized image coordinates'}
    save_json(root/'reports/B2_direction_summary.json',summary)
    # Fixed first six examples; no selection by apparent success or test direction label.
    examples = selected[:6]
    canvas = Image.new('RGB',(1200,len(examples)*380+28),'white');draw=ImageDraw.Draw(canvas)
    draw.text((8,8),'Original | B2 segmentation | Connected sidewalk and direction candidate',fill='black')
    for y,row in enumerate(examples):
        with Image.open(root/row['image_path']) as source:original=source.convert('RGB')
        mask=read_mask(root/'outputs/masks/C'/(row['image_id']+'.png'))
        color=np.full((*mask.shape,3),65,np.uint8)
        for cid in [1,2,8]:color[mask==cid]=COLORS[cid]
        with Image.open(out/(row['image_id']+'_direction.jpg')) as direction:direction=direction.convert('RGB')
        for x,pane in enumerate([original,Image.blend(original,Image.fromarray(color),.6),direction]):
            pane.thumbnail((400,330));canvas.paste(pane,(x*400,28+y*380))
        draw.text((8,28+y*380+338),row['original_path']+' | '+records[y]['direction_candidate'],fill='black')
    preview=root/'reports/B2_direction_examples.jpg';canvas.save(preview,quality=92)
    summary['preview']=str(preview);save_json(root/'reports/B2_direction_summary.json',summary)
    text=['# B2 보행 방향 후보 — 규칙 기반 Baseline','',f'고정 {split} 이미지: {len(records)}장',
          '인도 예측 중 화면 하단 중앙과 연결된 성분만 후보로 사용한다. 차도·자전거도로는 제외한다.',
          '4방향 격자 경로에서 위쪽·좌우 이동만 허용하고, 원본 해상도에서 경로가 후보 영역을 벗어나지 않는지 재검증한다.',
          '연결성·행별 상대 폭·영상 중심 편차·연결 통로 점수를 기록한다. 부족하면 No clear path(판단 보류)다.',
          '화살표는 영상 내 경로이며 실제 거리·통로 폭·안전 경로가 아니다.',
          '임계값은 선언된 기본값이며 방향 검증 라벨로 조정되지 않았다. 테스트 결과로 튜닝하지 않았다.',
          f'후보 분포: {summary["candidate_counts"]}',
          f'경로 계산 시간(초, 파일 읽기·시각화 제외): {summary["route_latency_seconds"]}',
          '사람이 지정한 방향 라벨이 없어 방향 정확도는 산출하지 않았다. 자동 출력을 정답으로 사용하지 않는다.',
          '3종 모델에는 기타·장애물·연석 분류가 없어 장애물 회피나 보행 안전을 보장하지 않는다.',
          'Decision Tree/Random Forest는 방향별 라벨 수 요건을 충족하지 않아 학습하지 않았다.',
          '', 'CSV: metrics/B2_direction_candidates.csv', '이미지와 JSON: outputs/b2_direction/',
          '미리보기: reports/B2_direction_examples.jpg']
    (root/'reports/B2_direction_report.md').write_text('\n'.join(text),encoding='utf8')
    return summary


class B2DirectionPredictor:
    """Load once; measure synchronized single-image FP32 inference separately from IO."""
    def __init__(self, root):
        import torch
        from transformers import SegformerForSemanticSegmentation
        self.root=Path(root); self.run,self.digest=_checkpoint(root)
        frozen=self.root/'data/splits/b2_route_rule_v1.json'
        self.config=load_json(frozen)['config'] if frozen.exists() else ROUTE_DEFAULTS
        self.device='cuda' if torch.cuda.is_available() else 'cpu'
        started=perf_counter()
        self.model=SegformerForSemanticSegmentation.from_pretrained(self.root/'checkpoints/C/best').to(self.device).eval()
        if self.model.config.num_labels!=3:raise ValueError('Three-class head required')
        self.sync(); self.load_seconds=perf_counter()-started

    def sync(self):
        import torch
        if self.device=='cuda':torch.cuda.synchronize()

    def predict(self, image):
        import torch
        image=image.convert('RGB'); self.sync(); started=perf_counter()
        pixels=np.asarray(image.resize((self.run['config']['size'],)*2,Image.Resampling.BILINEAR),dtype=np.float32)/255
        pixels=(pixels-np.array([.485,.456,.406],np.float32))/np.array([.229,.224,.225],np.float32)
        x=torch.from_numpy(pixels.transpose(2,0,1).copy())[None].to(self.device)
        self.sync(); prepared=perf_counter()
        with torch.inference_mode():
            logits=self.model(pixel_values=x).logits.float()
            local=torch.nn.functional.interpolate(logits,(image.height,image.width),mode='bilinear',align_corners=False).argmax(1)[0]
        mask=decode_task_mask(local.cpu().numpy(),[1,2,8]); self.sync(); inferred=perf_counter()
        result,connected=direction_candidate(mask,config=self.config); routed=perf_counter()
        result.update(source_model='SegFormer B2 C/best',checkpoint_sha256=self.digest,device=self.device,
                      timing_seconds={'preprocess':prepared-started,'inference_and_mask':inferred-prepared,
                                      'route':routed-inferred,'pipeline':routed-started})
        return result,mask,connected


def infer_b2_direction(root,image_path,output_path=None):
    predictor=B2DirectionPredictor(root)
    with Image.open(image_path) as source:image=source.convert('RGB')
    result,mask,connected=predictor.predict(image)
    rendered=render_candidate(image,mask,result,connected)
    if output_path:
        output_path=Path(output_path);output_path.parent.mkdir(parents=True,exist_ok=True)
        rendered.save(output_path);save_json(output_path.with_suffix('.json'),result)
    return result,mask,rendered


def benchmark_b2_direction(root,split='test'):
    """Fresh inference on the frozen set. No threshold tuning or direction accuracy."""
    import torch
    import pandas as pd
    import platform
    root=Path(root); rows=[r for r in validate_frozen(root) if r['split']==split]
    if not rows:raise ValueError('No images')
    predictor=B2DirectionPredictor(root)
    with Image.open(root/rows[0]['image_path']) as source:predictor.predict(source)  # unmeasured warmup
    if predictor.device=='cuda':torch.cuda.reset_peak_memory_stats()
    records=[]
    for i,row in enumerate(rows):
        with Image.open(root/row['image_path']) as source:image=source.convert('RGB')
        result,mask,connected=predictor.predict(image)
        cached=read_mask(root/'outputs/masks/C'/(row['image_id']+'.png'))
        valid=all(np.all(connected[yy,xx]) for a,b in zip(result['path_xy_normalized'],result['path_xy_normalized'][1:])
                  for yy,xx in [_line_pixels((round(a[1]*(mask.shape[0]-1)),round(a[0]*(mask.shape[1]-1))),
                                         (round(b[1]*(mask.shape[0]-1)),round(b[0]*(mask.shape[1]-1))))])
        records.append({'image_id':row['image_id'],'direction_candidate':result['direction'],
                        'path_inside_predicted_sidewalk':bool(valid) if result['path_xy_normalized'] else None,
                        'cached_mask_agreement':float(np.mean(mask==cached)),**result['timing_seconds']})
        if i==0:render_candidate(image,mask,result,connected).save(root/'reports/B2_direction_fresh_example.jpg',quality=92)
        if (i+1)%10==0:print(f'Fresh B2 + direction benchmark: {i+1}/{len(rows)}',flush=True)
    frame=pd.DataFrame(records);frame.to_csv(root/'metrics/B2_direction_timing.csv',index=False)
    summary={'images':len(rows),'split':split,'device':predictor.device,
             'hardware':torch.cuda.get_device_name() if predictor.device=='cuda' else platform.processor(),
             'torch':torch.__version__,'precision':'FP32','batch_size':1,'input_size':predictor.run['config']['size'],
             'warmup_images':1,'model_load_seconds':predictor.load_seconds,
             'timing_scope':'RGB image in memory through resize, model, original-resolution mask and route; excludes disk IO, model loading and rendering',
             'latency_seconds':{k:{'mean':float(frame[k].mean()),'median':float(frame[k].median()),'p95':float(frame[k].quantile(.95))} for k in ['preprocess','inference_and_mask','route','pipeline']},
             'candidate_counts':dict(Counter(frame.direction_candidate)),
             'no_clear_path_fraction':float((frame.direction_candidate=='No clear path').mean()),
             'paths_verified_inside_prediction':int(frame.path_inside_predicted_sidewalk.fillna(False).sum()),
             'paths_checked':int(frame.path_inside_predicted_sidewalk.notna().sum()),
             'minimum_cached_mask_agreement':float(frame.cached_mask_agreement.min()),
             'peak_cuda_allocated_mib':torch.cuda.max_memory_allocated()/2**20 if predictor.device=='cuda' else None,
             'direction_accuracy':None,'thresholds_tuned_on_test':False,'checkpoint_sha256':predictor.digest}
    save_json(root/'reports/B2_direction_timing_summary.json',summary)
    return summary
