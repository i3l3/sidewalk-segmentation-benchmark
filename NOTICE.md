# 출처와 이용 조건

이 저장소는 연구·수행평가용 코드입니다. `assets/exploration/`에는 초기 시각 비교의 재현에 사용한 사진 4장이 포함되어 있습니다. 출처와 파일 해시는 `exploration_photos.json`에 기록했습니다. 사진의 원저작권은 원출처에 있으며, 이 저장소의 코드 이용 조건이 사진의 별도 이용 허락을 부여하지는 않습니다. 학습용 이미지·정답 데이터셋·모델 가중치는 포함하지 않습니다.
모델 코드·가중치·학습 데이터의 이용 조건은 각각 확인해야 합니다.

- [Mask2Former 공식 저장소와 가중치 목록](https://github.com/facebookresearch/Mask2Former/blob/main/MODEL_ZOO.md): 공개 가중치의 CC BY-NC 4.0 조건 확인.
- [OneFormer 공식 구현](https://github.com/SHI-Labs/OneFormer): 원본 코드 MIT, 사전학습 backbone과 데이터 이용 조건도 별도 확인.
- [SegFormer 원본 라이선스](https://github.com/NVlabs/SegFormer/blob/master/LICENSE): NVIDIA 비상업적 연구·평가 조건 확인.
- [AI Hub 인도보행 영상](https://www.aihub.or.kr/aihubdata/data/view.do?aihubDataSe=data&currMenu=1&dataSetSn=189&topMenu=): 다운로드·사용 조건과 제공 설명서 확인.

`legacy/`의 어댑터는 공식 저장소의 고정 커밋을 내려받아 호출합니다. 해당 저장소의 저작권 고지와 라이선스를 유지하세요.
출력 방향은 영상 기반 후보이며 실제 보행 안전을 보장하는 안내가 아닙니다.
