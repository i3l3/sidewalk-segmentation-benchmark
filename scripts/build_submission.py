"""Build an offline-readable grading notebook from recorded experiment artifacts.

Images containing AI Hub samples are embedded only in the private submission copy.
No training, data review, or model evaluation is performed by this script.
"""
from pathlib import Path
import argparse
import base64
import contextlib
import io
import json
import shutil
import uuid
import nbformat as nbf

REPO = Path(__file__).resolve().parents[1]


def build(artifacts, destination):
    artifacts, destination = Path(artifacts), Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    load = lambda name: json.loads((artifacts/name).read_text(encoding='utf8'))
    completion = load('B0_B2_completion.json')
    timing = load('B2_direction_timing_summary.json')
    split = load('frozen_split_audit.json')
    assert completion['state'] == 'completed' and timing['images'] == 60
    assert split['actual'] == {'train':251,'validation':39,'test':60}
    a, c = completion['experiments']
    source_url = 'https://www.aihub.or.kr/aihubdata/data/view.do?aihubDataSe=data&currMenu=1&dataSetSn=189&topMenu='
    repository = 'https://github.com/i3l3/sidewalk-segmentation-benchmark'
    summary = f'''| 항목 | 실제 수행 내용 |
|---|---|
| 예측할 것 | 이미지 각 픽셀이 차도·인도·자전거도로 중 어디에 속하는지 |
| 입력 X / 정답 y | RGB 이미지 / 픽셀별 검수된 ID 마스크 |
| 데이터 | AI Hub 인도보행 영상 Surface_1.zip에서 350장 추출·검수 |
| 분할 | 학습 251장 / 검증 39장 / 고정 테스트 60장 |
| 학습 | SegFormer B0(A), B2(C), 같은 검수 라벨 사용 |
| 테스트 mIoU | B0 {a['test_miou']*100:.2f}% / B2 {c['test_miou']*100:.2f}% |
| 추가 구현 | B2 인도 예측에서 연결된 보행 방향 후보 계산 |
| 방향 처리 시간 | T4·FP32, 평균 {timing['latency_seconds']['pipeline']['mean']*1000:.1f} ms/장 |
| 미측정 항목 | 사람 방향 정답이 없어 방향 정확도 미측정 |'''
    data_table = '''| 속성·파일 | 내용과 의미 | 형태·범위 |
|---|---|---|
| image | 원본 RGB 영상의 한 프레임 | JPG, 1920×1080, 채널 3개, 0~255 |
| mask | 이미지와 같은 위치의 정답 클래스 | 단일 채널 PNG, 저장 ID 1·2·8·255 |
| image_id / image_path | 표본 식별자 / 이미지 상대 경로 | 문자열 |
| group_id / split | 촬영일 기반 그룹 / 학습·검증·테스트 구분 | 범주형 |
| status / label_verified | 라벨 검수 상태 / 확인 여부 | reviewed / true |
| image_sha / mask_sha | 검수한 파일이 이후 바뀌었는지 확인 | SHA-256 문자열 |

이미지와 정답 마스크가 학습 데이터이고, CSV는 파일 경로·분할·검수 기록을 관리하는 목록이다.
색상은 표시용이며 실제 손실 계산은 정수 클래스 ID로 한다.

| 클래스 | 의미 | 저장 ID → 학습 헤드 ID |
|---|---|---|
| 차도 | 일반 차도와 골목의 normal/speed_bump/damaged 영역 | 1 → 0 |
| 인도 | sidewalk로 주석된 영역 | 2 → 1 |
| 자전거도로 | bike_lane로 주석된 영역 | 8 → 2 |
| 학습·평가 제외 | 미표기, 점자블록, 주의/공사 구역, 횡단보도 속성 등 | 255 → 255 |

**분할 결과의 색:** 주황색=차도, 분홍색=인도, 파란색=자전거도로. 어둡게 표시된 정답 ignore 영역은 평가에서 제외한다.

미표기를 ‘기타’ 정답으로 취급하지 않는다. 기존 9종 저장 체계를 유지하되 현재 모델의 출력 헤드는 3개다.'''
    preprocessing = '''1. ZIP 전체를 해제하지 않고 XML 폴리곤이 지원되는 이미지에서 seed 42로 폴더별 분산 표본 350장을 준비했다. 원본 RGB 마스크의 색을 추측하지 않고 XML 폴리곤과 겹침 순서로 ID 마스크를 만들었다.
2. 육안 검수 피드백에서 골목의 차도 누락을 확인하여 사용자 확인 후 차도 정의에 골목을 포함했다. 350장의 라벨 검수 확인을 기록하고 파일 해시로 상태를 추적했다.
3. 이미지·마스크 크기, 허용 ID, 누락 파일, 전부 ignore인 표본과 중복을 검사한다. 라벨이 없는 픽셀은 추정해 채우지 않고 255로 제외한다.
4. 촬영일 식별자별로 묶고 dHash 거리 4 이하의 유사 사진 그룹을 합쳤다. 그룹을 나누지 않아 목표 250/50/50 대신 251/39/60이 됐다. 분할 간 동일 파일·dHash≤4 유사 쌍·그룹 중복은 각각 0건이었다. 실제 장소 독립성은 확인되지 않았다.
5. 학습 입력은 512×512로 변환하고 RGB를 ImageNet 평균·표준편차로 정규화한다. 이미지와 정답에 같은 기하 증강을 적용한다. 정답 리사이즈에는 nearest-neighbor만 사용하여 새 클래스 값이 생기지 않게 한다.
6. 수치형 표 데이터처럼 임의의 행·열을 삭제하거나 평균으로 결측치를 채우지 않는다. 모래로 덮인 인도 같은 어려운 장면도 실패 사례로 다루며, 테스트 결과가 나쁘다는 이유로 제거하지 않았다.'''
    results = '''| 모델 | 완료 epoch | Best epoch | 검증 mIoU | 테스트 mIoU | 테스트 Pixel Accuracy |
|---|---:|---:|---:|---:|---:|
'''
    for exp in completion['experiments']:
        results += f"| {exp['model']} | {exp['epochs_completed']} | {exp['best_epoch']} | {exp['best_validation_miou']*100:.2f}% | {exp['test_miou']*100:.2f}% | {exp['test_pixel_accuracy']*100:.2f}% |\n"
    class_results = '| 모델 | 차도 IoU | 인도 IoU | 자전거도로 IoU |\n|---|---:|---:|---:|\n'
    for model in ['B0','B2']:
        values = [r['iou'] for r in completion['class_metrics'] if r['model'] == model]
        class_results += '| '+model+' | '+' | '.join(f'{v*100:.2f}%' for v in values)+' |\n'
    interpretation = f'''B2는 같은 고정 테스트에서 B0보다 mIoU가 **{(c['test_miou']-a['test_miou'])*100:.2f}%p** 높았다. 차도 IoU는 **14.41%p**, 인도 IoU는 **0.67%p** 높았지만 자전거도로 IoU는 **0.30%p 낮았다**. 모든 클래스가 개선됐다고 해석하지 않는다.

설계상 모델 규모가 큰 B2의 표현력이 유리할 수 있다고 기대할 수 있으나 사전에 기록한 수치 목표는 없다. 실제로 전체 mIoU는 높았지만 자전거도로에는 이점이 나타나지 않았다.

이는 두 구조를 같은 자료로 학습한 비교이며, 동일 모델의 학습 전후 개선율이 아니다. 학습 전의 새 헤드는 무작위 초기화 상태였고 별도 성능 기준으로 측정하지 않았다.

검증 점수는 512×512, 테스트 점수는 원본 크기로 복원한 예측에서 계산했다. 검증-테스트 차이에는 분할 장면 차이·과적합 가능성·평가 해상도 차이가 함께 작용할 수 있어 원인을 하나로 단정할 수 없다.

자전거도로가 포함된 사진은 학습 19장·검증 3장·테스트 7장으로 적다. 인도 픽셀이 많아 Pixel Accuracy만 보면 자전거도로의 낮은 성능이 가려지므로 클래스별 IoU와 mIoU를 함께 본다. Boundary F1은 원본 해상도·허용 오차 2px 조건에서 B0 0.0096, B2 0.0247로 낮아 경계 품질도 추가 개선이 필요하다.'''
    directions = '''화면 하단 60%에서 인도만 후보로 사용하고, 하단 5%·중앙 20%와 연결된 성분을 찾는다. 차도·자전거도로는 후보에서 제외한다. 연결성, 영상 대비 폭, 중심 편차를 계산하고 격자에서 연결된 경로를 탐색한다. 원본 해상도에서도 경로 중심선이 예측 인도 밖으로 나가지 않는지 확인한다. 연결이 없거나 조건을 만족하지 못하면 `No clear path`(판단 보류)를 낸다.

| 방향 후보 | 왼쪽 | 직진 | 오른쪽 | 판단 보류 |
|---|---:|---:|---:|---:|
| 고정 테스트 60장 | 44 | 4 | 7 | 5 (8.3%) |

Tesla T4, FP32, 배치 1, 512×512 입력, 워밍업 1장 후 전처리→B2→원본 크기 마스크→방향 계산을 측정했다. 평균 **208.3ms**, 중앙값 **212.7ms**, 95백분위 **233.9ms**였다. 파일 읽기·모델 로딩·그림 저장 시간은 제외했다. 경로 계산만 평균 88.4ms, 최대 CUDA 할당 메모리는 약 558MiB였다. 이는 학습 시 FP16 메모리 측정과 다른 조건이다.

생성된 경로 55개 모두 **예측 인도** 안에 있었다. 이는 실제 정답 인도나 안전한 길임을 증명하지 않는다. 재추론 마스크는 저장된 B2 예측과 60장 모두 일치했다.

왼쪽 후보가 73.3%로 치우쳤다. 사람의 방향 정답이 없어 장면 분포와 규칙의 편향을 구분할 수 없으며, 이 테스트 결과로 규칙을 튜닝하지 않았다. 방향 정확도와 Decision Tree/Random Forest 학습 결과는 없다. 영상 좌표는 정규화 비율이며 미터 거리·실제 통로 폭이 아니다.'''
    limitations = '''- 350장·단일 seed의 소규모 실험이다. 촬영일 그룹을 나눴지만 실제 장소 독립성과 모든 영상의 보행자 시점은 확인되지 않았다.
- 현재 모델은 3종만 분류하므로 사람·차량·장애물·연석·점자블록·계단을 독립적으로 구분하지 못한다. 미표기 영역의 오분류는 현재 ignore 기반 정량 점수에 반영되지 않는다.
- 한국/비한국 보행자 그룹이 모두 확보되지 않아 국가별 비교와 2:1 가중 점수는 산출하지 않았다.
- 기존 5개 모델의 4장×5개 탐색 추론은 구현과 결과를 보존했다. 해당 4장에는 정답이 없어 정량 순위가 없으며, 이번 고정 테스트에서 5개 모델을 평가한 결과도 없다.
- B/D 자동 라벨 추가 학습, 9종 확장, 깊이·실제 거리 추정, 실제 보행 시스템 배포는 이번 완료 범위에 포함하지 않았다.'''
    conclusion = '''이번 실험에서는 B2가 B0보다 높은 테스트 mIoU를 보였고, 인도 예측을 연결된 방향 후보 계산으로 이어 붙일 수 있었다. 다만 자전거도로 성능과 경계 품질은 낮았으며, 방향 정답과 장애물 인식이 없어 실제 보행 안전 성능으로 해석할 수 없다.

후속 개선은 자전거도로·가려짐·모래 등 어려운 장면을 추가하고, 장소를 확인한 더 큰 분할과 여러 seed로 재평가하는 것이다. 사람의 방향 라벨을 별도로 수집한 뒤 검증 세트에서 규칙을 조정하고 새 고정 테스트로 평가할 수 있다. 점자블록·연석 등은 정답이 확보된 뒤 9종 확장으로 진행한다.'''
    readme = f'''# 수행평가 제출 자료 안내

1. **제출용_보행영역분할.ipynb**를 Colab에 열면 저장된 표·그래프·비교 이미지로 결과를 확인할 수 있다. 읽기에는 GPU, Drive 연결, 전체 실행이 필요 없다.
2. **수행평가_보고서.md**는 제공된 보고서 양식의 1~10번 항목을 따른다. 학번·이름과 9번 느낀 점은 본인이 작성한다.
3. **3분_발표안.md**는 실제 결과와 한계를 담은 발표 초안이다.
4. **코드_전체.md**와 **source.zip**에 재사용 모듈·테스트·실험 노트북을 함께 제공한다. 공개 코드: {repository}
5. `B0_B2_test_*.csv`, `B2_direction_*.csv`와 JSON은 계산 근거다. 학습 가중치·350장 데이터·분할 목록은 개인 Drive의 `Sidewalk-Segmentation-Benchmark/`에 보존한다.

## 실행 구분

- 채점용 노트북의 무거운 실행 플래그는 모두 False다. 결과 확인을 위해 재학습할 필요가 없다.
- 재현하려면 1번의 RUN_REPRODUCTION을 True로 하여 코드를 설치하고, 같은 Drive 데이터에 연결한 뒤 필요한 작업만 선택한다. 다른 사람의 계정에서는 원본 데이터·가중치에 대한 별도 접근 권한이 필요하다.
- 전체 개발 노트북은 `notebooks/finetuning.ipynb`로 보존했다. 초기 9종·A~D 확장 기능과 실제 완료한 3종 A/C를 혼동하지 않도록 채점용 사본을 별도로 만들었다.
- 실측 기록을 읽는 셀은 저장된 결과의 재표시이며 새 학습·평가를 수행하지 않는다.
'''
    report = f'''# 인공지능 프로젝트 보고서

학번: [본인 입력]  이름: [본인 입력]

## 1. 주제
보행 장면의 차도·인도·자전거도로 분할 모델 비교와 영상 기반 보행 가능 방향 후보 계산

## 2. 주제 설정 동기
기존에 개발하던 시각장애인 보행 지원 시스템의 보행 가능 구역 판정 모듈을 개선하고자 했다. 인도를 픽셀 단위로 구분한 뒤 화면 하단과 연결된 영역에서 방향 후보를 찾을 수 있는지 조사했다. 장치의 실제 안전 성능 검증까지 한 것은 아니다.

## 3. 데이터 수집
출처: [AI Hub 인도보행 영상]({source_url}), 서피스마스킹 `Surface_1.zip`. 사용자가 신청 후 내려받은 ZIP을 개인 Drive에 업로드했다. 원본에는 JPG·RGB 마스크·CVAT XML이 있으며, 9,558개 원본 이미지 중 지원되는 XML 폴리곤을 가진 후보에서 350장을 표본화했다. 원본 데이터는 별도로 확보해야 하며 공개 코드 저장소에 재배포하지 않는다.

## 4. 데이터 검토
{data_table}

학습/검증/테스트는 251/39/60장이다. 클래스 포함 사진 수는 차도 195/32/36장, 인도 248/39/60장, 자전거도로 19/3/7장이다. 한 이미지에 여러 클래스가 있어 합계는 전체 장수와 같지 않을 수 있다.

## 5. 데이터 전처리
{preprocessing}

## 6. 인공지능 모델 설계
사용 모델은 SegFormer B0와 B2다. 이미지의 세부 위치를 유지하면서 문맥을 사용하는 의미 분할 모델이고, 크기가 다른 두 모델을 같은 자료·설정으로 비교할 수 있어 선택했다. `nvidia/mit-b0`, `nvidia/mit-b2`의 ImageNet 사전학습 인코더에서 시작하여 새 3종 분할 헤드를 초기화해 학습했다.

독립 변수 X는 RGB 이미지(학습 시 3×512×512 정규화 실수 텐서), 종속 변수 y는 같은 위치의 픽셀 클래스(512×512 정수 ID)다. ignore=255 픽셀은 손실에서 제외한다. 추가 방향 규칙의 입력은 예측 마스크에서 얻은 연결성·상대 폭·중심 편차 등이고, 출력은 왼쪽·직진·오른쪽·판단 보류 후보다.

설정: seed 42, 최대 20epoch, AdamW, 학습률 6e-5, weight decay 0.01, 유효 배치 8(microbatch 2·누적 4), Cross-Entropy, CUDA FP16 AMP. 검증 mIoU가 가장 높은 체크포인트를 저장하고 patience 5로 조기 종료했다. 재개를 위해 optimizer·scheduler·AMP·난수·epoch 상태를 함께 저장했다.

## 7. 모델 성능 및 결과 분석
IoU=TP/(TP+FP+FN), mIoU는 3개 클래스 IoU의 산술 평균이다. Pixel Accuracy는 평가 대상 픽셀 중 정답 비율이다. ignore=255는 제외한다. 검증 세트로 Best 체크포인트를 선택하고 고정 테스트 60장에서 평가했다.

{results}

{class_results}

{interpretation}

### 보행 방향 후보와 처리 시간
{directions}

### 개선 방안과 적용 범위
{limitations}

## 8. 결론 및 모델 활용 방안
{conclusion}

## 9. 느낀 점 — 본인 작성
직접 겪은 경험과 생각을 아래 질문을 참고해 작성한다. 이 항목은 자동으로 완성하지 않았다.

- 데이터 다운로드·GPU 사용 제한·라벨 검수에서 실제로 가장 어려웠던 점은 무엇이었는가?
- 도로 누락을 발견하고 골목의 라벨 기준을 바꾸면서 무엇을 배웠는가?
- 전체 정확도와 클래스별 IoU를 함께 본 뒤 판단이 어떻게 달라졌는가?
- 실제 시스템에 연결하기 전에 무엇을 더 검증하고 싶은가?

## 10. 전체 코드
공개 저장소: {repository}

채점용 Colab 사본과 `코드_전체.md`, `source.zip`을 함께 제공한다. 핵심 코드는 `src/surface_import.py`, `src/dataset.py`, `src/training.py`, `src/evaluation.py`, `src/b2_direction.py`다. 전체 개발·재현용 노트북은 `notebooks/finetuning.ipynb`다. 공개 사전학습 모델·라이브러리와 AI 코딩 도구를 활용했으며, 제출자는 실행 결과와 설명을 직접 확인한다.
'''
    report = report.replace('### 보행 방향 후보와 처리 시간', '![학습 곡선과 모델 비교](B0_B2_comparison.png)\n\n![고정 테스트 비교](B0_B2_test_examples.jpg)\n\n### 보행 방향 후보와 처리 시간').replace('### 개선 방안과 적용 범위', '![방향 후보 비교](B2_direction_examples.jpg)\n\n### 개선 방안과 적용 범위')
    (destination/'수행평가_보고서.md').write_text(report,encoding='utf8')
    (destination/'읽어주세요.md').write_text(readme,encoding='utf8')
    speech = '''# 3분 발표 초안

## 0:00~0:30 — 주제와 예측 목표
기존에 만들던 시각장애인 보행 지원 시스템에서 보행 가능한 영역을 더 잘 구분하고 싶었습니다. 그래서 이미지 각 픽셀이 차도, 인도, 자전거도로 중 어디에 속하는지 예측하는 모델을 학습하고, 예측 인도에서 방향 후보를 계산했습니다.

## 0:30~1:10 — 데이터와 전처리
AI Hub 인도보행 영상의 이미지 350장과 XML 정답을 사용했습니다. 이미지가 입력 X이고 픽셀별 클래스 마스크가 정답 y입니다. 골목이 차도에서 빠진 문제는 검수 후 기준을 수정했습니다. 미표기 영역은 정답을 추측하지 않고 학습에서 제외했습니다. 비슷한 사진이 서로 다른 분할에 들어가지 않도록 촬영일과 유사 이미지 기준으로 묶어 학습 251장, 검증 39장, 테스트 60장으로 나눴습니다.

## 1:10~2:00 — 학습과 결과 [비교 그래프 제시]
ImageNet 인코더에 새 분할 헤드를 붙인 SegFormer B0와 B2를 같은 조건으로 학습했습니다. 검증 점수로 Best 모델을 고른 뒤 고정 테스트에서 평가했습니다. mIoU는 B0 57.74%, B2 62.66%였습니다. B2가 4.93%p 높았지만 자전거도로 IoU는 오히려 조금 낮았습니다. 자전거도로 사진이 학습 19장으로 적어서 전체 점수만 보고 성능이 좋다고 판단하면 안 된다는 점을 확인했습니다.

## 2:00~2:35 — 방향 후보 [원본·마스크·경로 이미지 제시]
B2의 인도 예측 중 화면 하단 중앙과 연결된 통로만 사용했습니다. 경로가 예측 인도를 벗어나면 제외합니다. 60장에서 왼쪽 44장, 직진 4장, 오른쪽 7장, 판단 보류 5장이 나왔고 T4에서 평균 208ms가 걸렸습니다. 사람이 정한 방향 정답이 없어서 이것을 방향 정확도라고 부르지는 않았습니다.

## 2:35~3:00 — 한계와 다음 단계
현재 3종 모델은 사람과 장애물을 별도로 구분하지 못하고 왼쪽 후보가 많이 나옵니다. 실제 안전한 길을 안내한다고 주장할 수는 없습니다. 다음에는 부족한 클래스와 어려운 장면을 늘리고, 방향 정답을 직접 검수해 검증하는 것이 필요합니다.

발표 전 표현과 속도를 본인 말투에 맞추고, 시간을 재며 연습한다.
'''
    (destination/'3분_발표안.md').write_text(speech,encoding='utf8')
    cells=[]
    def md(text): cells.append(nbf.v4.new_markdown_cell(text))
    def code(text, stdout=None):
        cell=nbf.v4.new_code_cell(text)
        if stdout is not None:cell.outputs=[nbf.v4.new_output('stream',name='stdout',text=stdout)]
        cells.append(cell)
    md('''**수행평가 제출용 — 보행 영역 분할과 방향 후보**

학번: [본인 입력]　이름: [본인 입력]

**읽는 방법:** 아래 저장된 표·그래프·비교 이미지를 위에서부터 확인하세요. 결과 확인에는 GPU·Drive 연결·전체 실행이 필요 없습니다. 실행 플래그는 모두 꺼져 있습니다. 코드 셀의 저장된 결과는 완료한 실험 기록이며 새 학습 결과가 아닙니다.

**구성:** ① 기존 모델 탐색과 환경 ② 데이터·정답·검수 ③ B0/B2 학습 ④ 고정 테스트·방향 후보·해석.

'''+summary)
    md('# 1. 기존 5개 모델 분할 코드\n\n기존 시스템의 보행 영역 판정 개선을 위해 분할 모델을 탐색했다. 초기 5개 모델은 사진 4장씩 총 20건의 탐색 추론을 수행했다. 정답 마스크가 없는 탐색 사진의 결과는 정량 순위로 사용하지 않았다. 최종 학습 비교는 아래 B0/B2 두 모델이다.')
    code('''# 선택 실행: 재현을 원할 때만 True. 결과 열람에는 실행할 필요 없음.
RUN_REPRODUCTION = False
import sys, subprocess, json
from pathlib import Path
REPO_URL = 'https://github.com/i3l3/sidewalk-segmentation-benchmark.git'
REPO_DIR = Path('/content/sidewalk-segmentation-benchmark')
ROOT = Path('/content/drive/MyDrive/Sidewalk-Segmentation-Benchmark')
if RUN_REPRODUCTION:
    if not REPO_DIR.exists():
        subprocess.run(['git','clone',REPO_URL,str(REPO_DIR)],check=True)
    subprocess.run([sys.executable,'-m','pip','install','-r',str(REPO_DIR/'requirements.txt')],check=True)
    from google.colab import drive
    drive.mount('/content/drive')
    sys.path.insert(0,str(REPO_DIR))
def run(command,*args):
    if not RUN_REPRODUCTION:
        raise RuntimeError('재현 설정을 먼저 실행하세요.')
    subprocess.run([sys.executable,'-m','src.runner',command,'--root',str(ROOT),*map(str,args)],cwd=REPO_DIR,check=True)
print('결과 열람 모드입니다. 학습·다운로드는 자동으로 실행하지 않습니다.')
''','결과 열람 모드입니다. 학습·다운로드는 자동으로 실행하지 않습니다.\n')
    md('''| 초기 모델 | 용도 |
|---|---|
| Mask2Former Swin-L | 탐색 분할 |
| OneFormer DiNAT-L | 탐색 분할 |
| Mask2Former R50 | 탐색 분할 |
| SegFormer B5 | 탐색 분할 |
| SegFormer B2 | 탐색 분할 |

기존 로더·시각화는 `legacy/`에 유지했다. 원본 구현의 의존성 충돌을 줄이기 위해 모델 실행은 독립 프로세스로 분리한다. 탐색 사진 4장과 예측 20개는 학습·검증·최종 테스트에 넣지 않았다. 다운로드·Tailscale 설정 같은 개발 과정은 재현용 노트북에 보존하고 채점용 본문에서는 생략했다.''')
    code('''RUN_LEGACY_EXPLORATION = False
if RUN_LEGACY_EXPLORATION:
    for model in ['mask2former_swin_l','oneformer_dinat_l','mask2former_r50','segformer_b5','segformer_b2']:
        # 기존 탐색 자료는 별도 root에 두어 학습 데이터와 분리
        subprocess.run([sys.executable,'-m','src.runner','benchmark','--root',str(ROOT/'exploration'),'--models',model],cwd=REPO_DIR,check=True)
''')
    md('# 2. 데이터셋 라벨링 코드\n\n**출처:** [AI Hub 인도보행 영상]('+source_url+'). `Surface_1.zip`에서 이미지 350장과 공식 XML 정답을 사용했다. 자동 Teacher 초안을 정답으로 간주한 실험이 아니며, 공식 주석을 현재 클래스에 맞춰 변환하고 사용자 검수를 거쳤다.\n\n'+data_table)
    md('## 전처리·검수·분할을 한 이유\n\n'+preprocessing)
    code('''# 재현 옵션: 기존 제출 데이터는 다시 만들지 않음. 새 경로에 별도 준비할 때만 사용.
RUN_NEW_DATA_PREPARATION = False
NEW_DATA_ROOT = Path('/content/sidewalk_new_data_version')
if RUN_NEW_DATA_PREPARATION:
    if not RUN_REPRODUCTION:raise RuntimeError('재현 설정을 먼저 실행하세요.')
    from src.surface_import import prepare_surface
    from src.schema import configure_task
    configure_task(NEW_DATA_ROOT,[1,2,8])
    prepare_surface(NEW_DATA_ROOT,ROOT/'data/downloads/aihub_189/manual/Surface_1.zip',count=350,seed=42)
    print('새 데이터의 육안 검수와 그룹 확인은 별도로 해야 합니다. 자동 승인하지 않습니다.')
''')
    md('''## 최종 데이터 점검 기록

| 구분 | 학습 | 검증 | 테스트 |
|---|---:|---:|---:|
| 전체 이미지 | 251 | 39 | 60 |
| 차도 포함 | 195 | 32 | 36 |
| 인도 포함 | 248 | 39 | 60 |
| 자전거도로 포함 | 19 | 3 | 7 |

350장 검수 완료. 파일 해시가 바뀌면 검수 상태를 해제한다. 파일·유사 사진·프록시 그룹의 분할 간 중복 검사는 0건이다. 촬영일 기준은 실제 장소의 완전한 독립을 보장하지 않는다. 분할 목록과 테스트 목록은 Drive `data/splits/`에 고정했다.''')
    md('# 3. 모델 학습 또는 파인튜닝 코드\n\n**실행 완료: A=B0, C=B2, 둘 다 같은 검수 라벨만 사용.**\n\n입력 X는 정규화 RGB 텐서, 정답 y는 대응하는 픽셀 클래스다. ImageNet 사전학습 인코더에 새 3종 분할 헤드를 붙였다. B0/B2는 모델 규모가 달라 같은 조건에서 성능과 자원 사용을 비교했다. 초기 새 헤드를 과거 학습 완료 모델로 취급하지 않는다.')
    md('''| 설정 | 실제 값 |
|---|---|
| seed / 입력 | 42 / 512×512 |
| optimizer / lr / weight decay | AdamW / 6e-5 / 0.01 |
| 손실 | Cross-Entropy, ignore=255 제외 |
| 유효 배치 / microbatch / 누적 | 8 / 2 / 4 |
| 최대 epoch / early stopping | 20 / 검증 mIoU patience 5 |
| GPU / 학습 정밀도 | Tesla T4 / FP16 AMP |
| 선택 기준 | 검증 mIoU가 가장 높은 Best 체크포인트 |

B0는 20epoch 중 19epoch를, B2는 조기 종료한 14epoch 중 9epoch를 Best로 선택했다. 테스트 점수로 학습 설정을 조정하지 않았다. 모델·optimizer·scheduler·AMP·난수·epoch를 저장하여 중단 후 재개할 수 있다. 별도 합성 자료로 저장→재개 흐름을 검증했고 합성 성능을 실제 성능에 섞지 않았다.''')
    code('''RUN_TRAIN_A_C = False  # 이미 완료했으므로 결과 확인 시 False 유지
TRAIN_CONFIG = {'seed':42,'size':512,'epochs':20,'lr':6e-5,'weight_decay':0.01,
                'effective_batch':8,'microbatch':2,'patience':5}
if RUN_TRAIN_A_C:
    for experiment in ['A','C']:
        run('train','--experiment',experiment,'--config',json.dumps(TRAIN_CONFIG))
''')
    md('핵심 학습 구현은 `src/training.py`의 `SegmentationDataset`, `safe_loss`, `train`이다. 검수 라벨·검증 세트가 없으면 정식 학습을 거부한다. 전체 파일은 함께 제공한 `코드_전체.md`와 `source.zip`에서 볼 수 있다. B/D와 9종 확장은 기능만 보존했으며 이번 실험 결과로 제시하지 않는다.')
    md('# 4. 성능 검증 코드\n\n검증 세트로 선택한 Best 체크포인트를 같은 고정 테스트 60장에서 평가했다. 정답 ignore=255를 제외한 누적 confusion matrix로 IoU·Precision·Recall·Pixel Accuracy를 구했다.\n\n**IoU** = TP/(TP+FP+FN), **mIoU** = 3개 클래스 IoU의 평균. 클래스 불균형 때문에 Pixel Accuracy만으로 모델을 판단하지 않는다.')
    code('''RUN_FIXED_TEST_EVALUATION = False
if RUN_FIXED_TEST_EVALUATION:
    for experiment in ['A','C']:
        run('evaluate','--experiment',experiment,'--split','test')
''')
    md('## 실제 테스트 결과\n\n'+results+'\n'+class_results+'\n'+interpretation)
    # Recorded evidence, directly from completed artifact JSONs (not re-evaluated here).
    snapshot={'provenance':'completed A/C experiment records; redisplay only',
              'experiments':completion['experiments'],'direction_timing':timing}
    result_code='''# 실측 JSON 스냅샷을 다시 표시한다. 새 학습·추론을 수행하는 셀이 아님.
RESULTS = json.loads('''+repr(json.dumps(snapshot,ensure_ascii=False))+''')
for record in RESULTS['experiments']:
    print(f"{record['model']}: test mIoU={record['test_miou']:.6f}, Pixel Accuracy={record['test_pixel_accuracy']:.6f}")
'''
    stdout=io.StringIO()
    with contextlib.redirect_stdout(stdout):exec(result_code,{'json':json})
    code(result_code,stdout.getvalue())
    image_specs=[]
    def image_cell(name,caption,private):
        md(caption)
        code("from IPython.display import display, Image, Markdown\n"+
             f"_name = {name!r}\n"+
             "_candidates = [Path(_name), ROOT/'reports'/_name, ROOT/'reports/graphs'/_name]\n"+
             "_file = next((p for p in _candidates if p.exists()), None)\n"+
             "if _file:display(Image(filename=str(_file)))\n"+
             "else:display(Markdown('저장된 그림은 제출용 사본에서 확인하세요. 재실행에는 결과 파일 경로가 필요합니다.'))\n")
        image_specs.append((len(cells)-1,name,private))
    image_cell('B0_B2_comparison.png','## 학습 곡선과 클래스별 성능\n\n실제 학습·검증 기록과 고정 테스트 지표를 시각화했다.',False)
    image_cell('B0_B2_test_examples.jpg','## 원본·정답·예측·오류·연석 확대 비교\n\n고정 테스트 목록의 처음 6장이다. 결과가 좋아 보이는 사진을 골라 넣지 않았다. ignore 영역은 어둡게 표시한다. 연석은 현재 독립 학습 클래스가 아니므로 확대 이미지는 주변 경계 관찰용이다.',True)
    md('## B2 보행 방향 후보와 처리 시간\n\n'+directions)
    code('''RUN_B2_DIRECTION = False
RUN_B2_TIMING = False
DIRECTION_IMAGE_PATH = ''  # 새 사진 파일 경로
if RUN_B2_DIRECTION or RUN_B2_TIMING or DIRECTION_IMAGE_PATH:
    if not RUN_REPRODUCTION:raise RuntimeError('재현 설정을 먼저 실행하세요.')
    from src.b2_direction import run_b2_direction, benchmark_b2_direction, infer_b2_direction
    if RUN_B2_DIRECTION:display(run_b2_direction(ROOT,'test'))
    if RUN_B2_TIMING:display(benchmark_b2_direction(ROOT,'test'))
    if DIRECTION_IMAGE_PATH:
        result,mask,preview=infer_b2_direction(ROOT,DIRECTION_IMAGE_PATH)
        display(preview)
        print(result)
''')
    image_cell('B2_direction_examples.jpg','## 방향 후보 예시\n\n원본 → B2 분할 → 하단 연결 영역과 경로. 초록색은 연결된 인도 예측, 노란 선은 경로 중심선이며 실제 안전 판정이 아니다.',True)
    md('## 결론·활용·개선 방안\n\n'+conclusion+'\n\n'+limitations)
    md('''## 제출 전 본인이 채울 항목

- 첫 셀과 보고서의 **학번·이름**을 입력한다.
- 보고서 9번 **느낀 점**은 실제 경험을 바탕으로 직접 작성한다.
- 제출할 때 파일을 다시 열어 표·그래프·비교 이미지가 보이는지 확인한다.

제공된 보고서 양식의 1~10번 내용은 `수행평가_보고서.md`, 발표 초안은 `3분_발표안.md`에 정리했다. 결과 자료·코드 출처를 표시하고, 방향 정확도나 실사용 안전성을 측정한 것처럼 작성하지 않는다.''')
    nb=nbf.v4.new_notebook(cells=cells,metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python'},'colab':{'name':'제출용_보행영역분할.ipynb'},'submission':{'mode':'recorded_results_read_only_by_default','actual_experiments':['A','C'],'active_canonical_ids':[1,2,8],'test_images':60}})
    public=nbf.from_dict(json.loads(nbf.writes(nb)))
    for index,name,private in image_specs:
        path=artifacts/name
        mime='image/png' if path.suffix=='.png' else 'image/jpeg'
        output=nbf.v4.new_output('display_data',data={mime:base64.b64encode(path.read_bytes()).decode(),'text/plain':name})
        nb.cells[index].outputs=[output]
        if not private:public.cells[index].outputs=[output]
        shutil.copyfile(path,destination/name)
    nbf.validate(nb);nbf.validate(public)
    nbf.write(nb,destination/'제출용_보행영역분할.ipynb')
    nbf.write(public,REPO/'notebooks/submission.ipynb')
    for name in ['B0_B2_completion.json','B0_B2_test_summary.csv','B0_B2_test_class_metrics.csv','B2_direction_candidates.csv','B2_direction_timing.csv','B2_direction_timing_summary.json','b2_route_rule_v1.json']:
        shutil.copyfile(artifacts/name,destination/name)
    full=['# 전체 코드 — 제출 시 열람용\n\n각 파일은 원래 모듈 구조로 실행한다. 이 문서 전체를 하나의 셀로 실행하지 않는다.\n']
    for pattern in ['src/*.py','legacy/*.py','scripts/*.py','tests/*.py']:
        for path in sorted(REPO.glob(pattern)):
            if path.resolve()==Path(__file__).resolve():continue
            relative=path.relative_to(REPO)
            full.extend([f'\n## {relative.as_posix()}\n\n```python\n',path.read_text(encoding='utf8'),'\n```\n'])
    (destination/'코드_전체.md').write_text(''.join(full),encoding='utf8')
    (REPO/'SUBMISSION.md').write_text(readme,encoding='utf8')
    print(json.dumps({'submission_notebook':str(destination/'제출용_보행영역분할.ipynb'),'cells':len(nb.cells),'four_sections':True,'embedded_evidence_images':len(image_specs),'public_notebook_excludes_aihub_sample_images':True},ensure_ascii=True))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--artifacts',type=Path,required=True)
    parser.add_argument('--destination',type=Path,required=True)
    args=parser.parse_args();build(args.artifacts,args.destination)
