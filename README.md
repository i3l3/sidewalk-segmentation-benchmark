# 보행 지원 시스템 2차 개발

`notebooks/benchmark.ipynb`의 최상위 구역은 요청한 네 부분입니다. `notebooks/finetuning.ipynb`는 같은 내용을 담은 이름 별칭입니다.

1. **기존 5개 모델 분할 코드**: 설치·GPU·Drive 설정, 기존 모델별 독립 프로세스 추론.
2. **데이터셋 라벨링 코드**: 자료 가져오기, 장소/시퀀스 분할, Teacher 2종 초안, 외부 검수, Teacher 선택.
3. **모델 학습 또는 파인튜닝 코드**: ImageNet 인코더에서 B0/B2 A~D 실험, best 저장, epoch 단위 재개.
4. **성능 검증 코드**: 고정 테스트 평가, 오류·연석 확대, 그래프, 보행 영역, 방향 후보와 방향 분류기, 보고서.

기존 모델 로더는 `legacy/`에서 재사용합니다. 1차 노트북과 사진 4장·20개 실험 결과는 이전 배포 자료에 보존되어 있으며, 이 공개 저장소에는 원본 사진·정답·가중치를 넣지 않았습니다.

[Colab에서 열기](https://colab.research.google.com/github/i3l3/sidewalk-segmentation-benchmark/blob/main/notebooks/benchmark.ipynb)

## GitHub 기반 실행

노트북 첫 구역에서 이 저장소를 `/content/sidewalk-segmentation-benchmark`에 clone합니다.
노트북에 인코딩한 코드 묶음을 넣지 않으며, `src/`와 `legacy/`의 코드를 직접 사용합니다.
이미지·라벨·체크포인트·결과는 설정한 Google Drive `ROOT`에 저장합니다.

- `REPO_REF='main'`: 현재 main 커밋을 가져옵니다.
- 재현 실험: 출력된 SHA를 `REPO_REF`에 지정해 같은 코드로 실행합니다.
- 실제 커밋은 `ROOT/metrics/source_revision.json`에 기록됩니다.
- clone 폴더에 수정 사항이 있으면 덮어쓰지 않고 중단합니다. 수정본을 보존하거나 다른 `REPO_DIR`를 지정하세요.
- 실행 결과·데이터·비밀값은 `.gitignore`로 저장소에서 제외합니다.

로컬에서 기본 검사:

```bash
python -m pip install numpy scipy pandas Pillow scikit-learn matplotlib joblib nbformat
python -m unittest discover -s tests -v
python scripts/validate_notebooks.py
```

## 현재 완료 상태

- 실제 학습용 이미지, 사람이 검수한 정답, 방향 정답은 아직 없습니다.
- 데이터·분할·검수·학습·평가 코드를 구현했고, 합성 자료의 작은 모델로 저장·재개 흐름을 검증했습니다.
- 합성 테스트 점수는 실제 B0/B2 또는 한국 보행 환경 성능이 아닙니다. 최종 모델이나 개선 정도를 아직 확정하지 않았습니다.
- 공식 mit-b0/mit-b2 인코더와 새 9종 분할 헤드의 로딩·출력·유한 손실을 Colab에서 확인했습니다.

## 실제 실험을 시작하는 순서

1. Colab GPU를 선택하고 노트북의 설치·Drive 셀을 실행합니다.
2. [AI Hub 인도보행 영상](https://www.aihub.or.kr/aihubdata/data/view.do?aihubDataSe=data&currMenu=1&dataSetSn=189&topMenu=)의 이용 조건을 확인하고 필요한 이미지 자료를 다운로드합니다. 인증은 사용자가 진행합니다.
3. 폴더 또는 ZIP 경로를 `SOURCE_IMAGES`에 지정합니다. 기본 수집 상한은 중복 제외 350장입니다. 원본 경로와 그룹 힌트를 CSV에 남깁니다.
4. `data/dataset_metadata.csv`에서 `group_id`를 촬영 장소·시퀀스 단위로 채웁니다. 연속 프레임은 같은 그룹이어야 합니다. `camera_view=pedestrian`은 확인한 사진에만 입력합니다.
5. `CONFIRM_GROUPS=True`로 train/validation/test를 고정합니다. 목표는 250/50/50이며 그룹 보존 때문에 실제 수량은 달라질 수 있습니다. 이후 이미지를 추가하려면 새 데이터 버전 경로를 사용합니다.
6. 두 Teacher의 초안을 생성하고 검수 ZIP을 내보냅니다. `draft_masks/`와 `overlays/`를 구분합니다.
7. 외부 라벨링 도구에서 검증·테스트 전부와 학습 일부를 검수합니다. 예를 들어 train 50장 검수 + 나머지 200장 자동 라벨이면 A/C와 B/D를 비교할 수 있습니다. 이 숫자는 권장 초기 구성이며 실제 확보량을 기록해야 합니다.
8. 수정한 **ID PNG만 있는 폴더**를 `REVIEWED_MASKS`로 지정하고, 검수자와 `CONFIRM_REVIEWED=True`를 입력합니다. 파일명은 `image_id.png`, 크기는 원본과 같아야 합니다.
9. `SELECT_TEACHER=True`로 검수된 train/validation 자료에서만 Teacher를 선택합니다. test는 선택에 사용하지 않습니다.
10. 실제 자료가 준비되면 `RUN_REAL_TRAINING=True`. A/C에는 검수 학습 라벨, B/D에는 그 라벨과 별도 이미지의 Teacher 초안이 필요합니다.
11. validation으로 설정·best checkpoint를 선택한 뒤 `RUN_FINAL_EVALUATION=True`로 고정 테스트 평가를 실행합니다.
12. 방향 정답을 원본을 보고 직접 입력한 다음 `RUN_DIRECTION=True`로 특징·방향 후보를 계산합니다.

기존 4장 탐색 사진은 파일/픽셀 해시로 학습 자료에서 제외합니다. 비슷한 사진 검사는 64bit dHash 거리 4 이하를 같은 분할 그룹으로 묶습니다. 촬영 시퀀스 분리를 대체하는 검사는 아닙니다.

## 라벨 정의와 외부 검수

| 학습 ID | 클래스 | 기존 공통 ID |
|---|---|---|
| 0 | Other | 0 |
| 1 | Road | 1 |
| 2 | Sidewalk | 2 |
| 3 | Curb | 3 |
| 4 | Crosswalk (plain) | 4 |
| 5 | Pedestrian Area | 7 |
| 6 | Curb Cut | 8 |
| 7 | Crosswalk Marking | 9 |
| 8 | Bike Lane | 10 |
| 255 | Ignore | 5, 6, 11, 65535 |

점자블록·계단·주의구역은 이번 학습 체계에서 제외합니다. 빈 미라벨 영역을 Other로 채우지 않습니다. Curb Cut, Crosswalk Marking, Pedestrian Area를 다른 클래스로 합치지 않습니다.

`src.labeling.import_mask`는 명시적인 RGB→ID 매핑을 제공했을 때만 색상 마스크를 변환합니다. 검수 후 이미지 또는 마스크가 바뀌면 검수 승인이 해제됩니다. 검수하지 않은 초안은 `pseudo`, 검수 완료는 `reviewed`, 제외 대상은 `rejected`입니다.

AI Hub 공식 노면 폴리곤을 가져올 때는 배포본 설명서의 라벨·속성 체계를 확인해야 합니다. `import_aihub_polygons` 입력은 명시적인 다음 중간 형식입니다.

```json
{"polygons": [{"label": "공식 라벨과 속성의 명시적 조합", "points": [[10,20],[50,20],[50,60]]}]}
```

공식 배포본의 실제 JSON은 아직 확보하지 않았으므로 자동으로 스키마를 추측하지 않습니다. 해당 형식으로 정규화하고 `label_mapping`을 제공하거나, ID PNG를 가져오세요. 미확인 라벨은 ignore로 남습니다. 공식 라벨도 자동으로 수동 검수 완료로 표시하지 않습니다.

## 학습과 재개

- A/B: `nvidia/mit-b0`, C/D: `nvidia/mit-b2`. 처음 선택한 모델 리비전을 저장해 실험 간 같은 인코더를 사용합니다.
- 학습 분할 헤드는 새로 초기화되며 과거 학습 B0가 아닙니다.
- 기본 seed42 / 512×512 / AdamW lr6e-5 / weight_decay .01 / effective batch8 / max20epoch / patience5.
- 이미지와 마스크에 같은 crop/flip을 적용하고, 이미지에만 밝기 증강을 적용합니다. 마스크 리사이즈는 nearest-neighbor입니다.
- GPU 메모리에 따라 microbatch를 설정하고 gradient accumulation으로 유효 배치를 유지합니다. 메모리 부족 시 `microbatch=1`로 새 실행 설정을 지정하세요.
- `latest.pt`에는 모델·optimizer·scheduler·AMP·RNG·완료 epoch를 저장합니다. `best/`에는 validation mIoU가 가장 높은 safetensors를 저장합니다.
- 설정·분할·이미지/라벨 해시가 바뀌면 기존 체크포인트에서 재개하지 않습니다. 마지막 완료 epoch부터 재개하며 미완료 epoch는 다시 실행합니다.

## 평가·보행 방향

- 검증·테스트 정답은 전부 사람이 검수한 라벨이어야 합니다.
- 미지원 클래스는 N/A. 공통 Other/Road/Sidewalk mIoU를 별도로 기록합니다. 지원 범위가 다른 전체 mIoU나 Pixel Accuracy만으로 순위를 정하지 않습니다.
- Boundary F1의 허용 오차는 원본 2px입니다. GT ignore 인접 경계는 제외합니다.
- 한국/비한국 보행자 그룹이 모두 있어야 2:1 보조 점수를 계산합니다. 같은 이미지는 두 그룹에 중복 집계하지 않습니다.
- 보행 후보는 인도·보행자 구역이며 하단 중앙과 연결된 성분만 남깁니다. 면적·행별 폭·연속성·중심 편차는 모두 영상 상대값입니다.
- 장애물 마스크가 별도로 없으면 obstacle_ratio는 결측입니다. 거리를 미터로 추정하지 않고 결과를 안전 경로라고 표현하지 않습니다.
- 방향 정답은 `set_direction_label(..., reviewer='실제 검수자', confirmed=True)`로 직접 지정합니다.
- 방향별 train20/validation5/test5 이상일 때만 Decision Tree·Random Forest를 학습합니다. 그렇지 않으면 규칙 Baseline만 제공합니다.
- 각 분할 모델의 특징으로 분류기를 따로 학습·평가해 모델 사이 특징 분포가 섞이지 않게 합니다.

## 주요 출력

- `data/dataset_metadata.csv`, `data/splits/locked.json`, `data/class_schema.json`
- `metrics/review_priority.csv`, `metrics/teacher_selection.json`
- `checkpoints/A~D/{run.json,history.json,latest.pt,best/}`
- `metrics/segmentation_metrics.csv`, 모델별 클래스 지표·공통 지표·국가 그룹 지표
- `metrics/direction_features.csv`, 방향별 분류 지표·confusion matrix
- `outputs/comparisons/`: 원본·정답·예측·오류와 연석 확대
- `outputs/direction_predictions/`: 연결 마스크와 방향 후보 이미지
- `reports/graphs/`, `reports/experiment_report.md`

실제 자료가 없는 경우 결과표를 채우지 않습니다. `validation.json`은 소프트웨어 동작 검증 기록이고 모델 성능 결과가 아닙니다.


## AI Hub 선택 다운로드

노트북 2번 구역의 **AI Hub 공식 도구로 선택 다운로드**를 사용한다.
1. AI Hub 인도보행 영상(189) 페이지에서 다운로드 이용 신청·승인을 받고 API 키를 발급받는다.
2. Colab Secrets에 `AIHUB_API_KEY`를 등록하고 노트북 접근을 허용한다. 없으면 숨김 입력을 사용한다.
3. `RUN_AIHUB_LIST=True`로 파일 목록을 조회하고 원천 이미지·대응 라벨 ZIP의 `file_key`를 확인한다.
4. `AIHUB_FILE_KEYS=['파일키', ...]`를 채우고 `RUN_AIHUB_DOWNLOAD=True`로 실행한다. 전체 일괄 다운로드는 허용하지 않는다.
5. 출력된 원천 ZIP 경로를 `SOURCE_IMAGES`에 지정해 기존 가져오기 셀을 실행한다. 여러 원천 ZIP은 각각 가져온다. 라벨 ZIP은 따로 보관하고 실제 스키마와 9종 변환표를 확인한 뒤 가져온다.

`src/aihub_download.py`는 공식 `aihubshell`을 Linux/Colab에서 내려받아 구문 검사하고 출처·SHA256을 기록한다. 키는 프로세스 환경으로 전달하며 명령 인자·저장소에 넣지 않는다. 선택 용량의 3배 여유 공간을 검사하고 독립 폴더에 다운로드한다. 공식 도구의 종료 코드만 믿지 않고 HTTP 성공 표시·선택 아카이브 존재·ZIP 구조까지 확인한다. 실패한 다운로드는 완료로 기록하지 않는다. 로그에는 키를 가린다. 이미 있는 다운로드 폴더는 덮어쓰지 않는다. 재개 다운로드는 이 래퍼에서 보장하지 않는다.

공식 도구가 전송 TAR를 풀고 분할 파일을 병합하더라도 내부 ZIP은 그대로 남는다. 데이터 다운로드는 검수나 학습 완료를 의미하지 않는다. 출처: [AI Hub 공식 aihubshell 안내](https://aihub.or.kr/devsport/apishell/list.do).


## Colab Tailscale exit node

노트북 상단 Git clone 셀 바로 뒤에 선택 연결 셀을 제공한다. `RUN_TAILSCALE=True`로 설치·로그인을 실행하고 출력 링크에서 기존 exit node와 같은 tailnet에 로그인한다. 다음 셀에서 승인된 온라인 exit node 목록을 확인한 뒤 `TAILSCALE_EXIT_NODE`에 국내 노드 이름이나 IP를 입력하고 다시 실행한다. 연결 성공 시 프록시의 공인 IP를 표시하고 `AIHUB_PROXY_URL`을 현재 커널 메모리에 설정한다. 이 IP가 국내 exit node의 공인 IP인지 확인한다.

`src/tailscale_proxy.py`는 공식 static 패키지의 SHA256을 확인한 뒤 `/content/sidewalk-tailscale`에서 userspace HTTP 프록시를 실행한다. 인증 상태는 RAM에만 유지하며 Drive나 저장소에 기록하지 않는다. AI Hub 모듈이 실행하는 curl과 도구 설치 요청에만 프록시를 전달한다. 전역 `HTTPS_PROXY`, 시스템 라우팅, Git/Hugging Face/Drive 경로는 바꾸지 않는다. `DISCONNECT_TAILSCALE=True`로 해제할 수 있다. 국내 IP 경유는 AI Hub 인증이나 이용 승인을 대신하지 않는다. 실제 인증된 exit-node 다운로드는 사용자의 로그인·노드 선택 후 별도로 검증한다.

원본 AI Hub ZIP 기본 저장 위치는 `/content/aihub_downloads`로 변경했다. 런타임 삭제 시 사라지므로 선별 데이터는 Drive에 별도 보관한다.


## 원본 데이터셋 저장 위치 선택

다운로드 설정 셀의 `SAVE_DATASET_TO_DRIVE`로 원본 ZIP의 저장 위치를 선택한다. 기본값 `False`는 `/content/aihub_downloads`이며, `True`는 `/content/drive/MyDrive/Sidewalk-Segmentation-Benchmark/data/downloads/aihub_189`이다. Drive를 선택하면 필요한 경우 마운트한다. 이 변수는 학습 결과 저장 설정인 `USE_DRIVE`와 독립적이다. 설정 셀을 다시 실행한 뒤 다운로드를 시작한다. 진행 중인 다운로드나 기존 파일은 자동 이동하지 않는다.


## 직접 받은 Surface ZIP을 작은 표본으로 업로드

확인된 사용자 ZIP 목록은 `Surface_NNN/*.jpg` 9,558개, `Surface_NNN/MASK/*.png` 9,558개, 폴더별 XML 130개이다. PNG는 마스크 위치로 분리하며 실제 ID/색상 의미는 별도 확인한다. 이미지 가져오기는 `MASK`/`masks` 하위 경로를 제외한다. 공식 마스크를 사람이 검수한 라벨로 자동 승격하지 않는다.

`python3 scripts/prepare_surface_subset.py --source Surface_1.zip --output sidewalk_subset_350.zip --count 350`으로 폴더에 분산된 JPG·대응 PNG 350쌍과 관련 XML을 묶는다. 원본 ZIP 전체를 풀지 않으며 원본 마스크·XML 값을 보존한다. 실제 선별 용량, PNG 헤더, XML 클래스 이름(지원 형식인 경우), 파일 해시는 `subset_manifest.json`에 기록한다. `Surface_NNN`은 그룹 후보일 뿐 실제 장소·시퀀스 확인을 대신하지 않는다.

생성한 ZIP을 `내 드라이브/Sidewalk-Segmentation-Benchmark/data/uploads/sidewalk_subset_350.zip`에 올리고 `SOURCE_IMAGES='/content/drive/MyDrive/Sidewalk-Segmentation-Benchmark/data/uploads/sidewalk_subset_350.zip'`로 지정한다. 마스크와 XML은 가져오기 캐시에 보존되며 실제 클래스 정의·색상/ID 매핑을 검증한 뒤 라벨 변환을 연결한다. 변환 확인 전 정식 학습은 실행하지 않는다.


## Surface XML 실제 학습 ID 변환

`src/surface_import.py`와 노트북 2번 구역의 `RUN_SURFACE_PREPARATION` 셀로 전체 ZIP 압축 해제 없이 350개 표본을 준비한다. XML 폴리곤 z_order를 보존하고 sidewalk→2, roadway의 normal→1, bike_lane→8로만 변환한다. alley, braille_guide_blocks, caution_zone, roadway crosswalk와 미표기는 255이다. RGB 색상은 추측하지 않는다. 원본 RGB PNG, XML, 원본 클래스 ID 마스크와 속성을 따로 보존한다. polyline/box 등 미지원 도형이 있는 이미지는 제외한다.

지원 라벨이 있는 이미지에서 폴더별 round-robin으로 선별하므로 선택 편향이 있으며, 9종 전체 성능을 검증할 수 없다. 선택 정책과 클래스 픽셀 수는 `data/surface_preparation.json`에 저장한다. 검수 상태는 `needs_review`, group_id와 split은 미확정으로 둔다. `data/review/surface_review.csv`와 surface_overlays에서 검수한다. 공식 마스크는 자동으로 reviewed가 되지 않는다. 검수·장소 분할 없이 정식 학습하지 않는다. 중단 후 동일 설정으로 재실행하면 기존 파일 해시를 검사하고 이어간다.


## 현재 실험은 3종

현재 노트북은 `configure_task(ROOT, [1,2,8])`로 차도·인도·자전거도로를 선택한다. 모델 출력 헤드는 실제 3개이며 내부 학습 ID는 0=차도, 1=인도, 2=자전거도로다. 원본/저장/평가 마스크의 canonical ID는 1/2/8을 유지한다. 학습에서만 연속 ID로 변환하고 추론 결과는 canonical ID로 복원하므로 기존 방향 특징 코드도 유지된다. 활성 클래스 밖의 라벨은 손실과 정량 평가에서 ignore로 처리하며 9종 성능으로 보고하지 않는다. 설정과 클래스 매핑은 체크포인트 서명에 포함한다. 분할 고정 또는 학습 이후에는 task 변경을 차단한다. 9종으로 확장할 때는 원본에서 라벨을 추가하고 새 데이터 버전·새 출력 헤드/체크포인트를 만든다. 3종 체크포인트의 출력 의미를 9종으로 재해석하지 않는다.
