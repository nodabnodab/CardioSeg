# 🫀 CardioSeg3D 개발 진행 로그 (Notion 업로드용)

본 문서는 **3D Cardiac MRI 다중 구조 분할 및 임상 지표 정량화 파이프라인(CardioSeg3D)** 프로젝트의 진행 과정, 기술적 도전 과제, 해결 방식 및 작업 현황을 기록하는 노션(Notion) 관리용 개발 로그입니다.

---

## 📅 프로젝트 개요
* **프로젝트명**: CardioSeg3D (카디오세그3D)
* **목표**: 3D Cine-MRI 데이터를 활용하여 우심실(RV), 좌심실(LV), 심근(MYO)을 3D U-Net으로 분할하고, 박출률(EF) 및 심근 질량(Mass)을 자동으로 계산하는 엔드투엔드 파이프라인 구축.
* **주요 개발 환경**: Windows OS / NVIDIA RTX 4070 Ti (12GB VRAM) / Python 3.11.9
* **핵심 라이브러리**: PyTorch (CUDA 12.1), MONAI, SimpleITK, Nibabel, Streamlit

---

## 🏆 마일스톤 및 진행 현황

### ✅ [완료] Phase 1: 데이터 확보 및 메타데이터 파싱 (1일차)
* **작업 내용**:
  * MICCAI 챌린지 공식 **ACDC 데이터셋** 확보 및 로컬 구조화 (`data/ACDC`)
  * `inspect_data.py` 작성을 통한 3D NIfTI(`.nii.gz`) 원본 이미지 헤더 분석.
* **기술적 획득**:
  * 환자마다 촬영 해상도(Voxel Spacing)가 상이함을 확인 (예: `1.5625 x 1.5625 x 10.0 mm`).
  * 복셀 물리 간격이 부피 계산(`ml`) 및 3D 일관성 유지의 핵심임을 도출.

### ✅ [완료] Phase 2: 대화형 프로토타입 대시보드 구축
* **작업 내용**:
  * 비의학 전공자 및 판독의 소통용 **Streamlit 가이드 웹 앱(`app_guide.py`)** 구현.
  * 심장 3대 구조(RV, MYO, LV) 설명 카드 및 박출률(EF) 공식 LaTeX 렌더링.
  * **실시간 복셀 부피 계산 샌드박스** 구축 (Spacing 수치와 복셀 수 입력 시 환자의 심실 부피를 실시간 계산하여 임상 수치 납득).

### ✅ [완료] Phase 3: 3D 전처리 파이프라인 구축 및 대비 최적화 (2일차)
* **작업 내용**:
  * MONAI 딕셔너리 트랜스폼 기반 3D 전처리 코드(`visualize_preprocessing.py`, `visualize_tuning.py`) 개발.
  * 해상도 정렬(`Spacingd` ➡️ `1.25 x 1.25 x 5.0 mm`), 빈 공간 절단(`CropForegroundd`), 밝기 정준화(`NormalizeIntensityd`) 적용.
* **🔥 기술적 이슈 및 해결 (Troubleshooting)**:
  * **문제점**: 밝기를 정규화(Z-score)한 뒤 전체 범위로 시각화했을 때, 극단적인 밝기 값의 노이즈들 때문에 실제 심장 벽의 명암비가 죽어 이미지가 극도로 흐릿하게(Blurry/Washed-out) 보이는 현상 발생.
  * **분석**: PyTorch의 3D 공간 보간은 `bilinear`(Trilinear)와 `nearest`만 연산 가능하여 물리적인 업샘플링 보간 흐림도 동반됨.
  * **해결**: 의료 영상 전처리 표준인 **Contrast Windowing (상/하위 2% 클리핑)** 기법을 시각화에 적용. `[percentile(2), percentile(98)]` 밖의 노이즈 밝기 값을 잘라냄으로써 심실 벽과 혈류 영역의 흑백 경계를 뚜렷하게 복원 완료.
  * **Streamlit 연동**: 웹 대시보드에 '일반 시각화' vs '대비 개선 시각화' 토글 기능을 추가하여 선명도 변화를 즉시 확인 가능하도록 개선.

### ✅ [완료] Phase 4: 환자 단위 3분할 데이터셋 구축 & 대비 정량 필터 연동 (2일차)
* **작업 내용**:
  * 데이터 누수(Data Leakage)를 원천 차단하기 위해 환자 번호 기준(Patient-level)으로 **Train(80명, 160파일) / Val(10명, 20파일) / Test(10명, 20파일)**로 격리 완료 (`src/dataset.py`).
  * 3D 데이터셋 파이프라인에 대비 개선(Contrast Windowing) 필터인 `ScaleIntensityRangePercentilesd`를 공식 연동하여, 로딩되는 모든 3D 이미지의 강도를 `0.0 ~ 1.0` 표준 스케일로 균일하게 자동 가공.
  * `verify_dataset.py`를 통해 실제 MONAI Dataset 로딩 시, 이미지 세기 범위가 정확히 최솟값 0.0, 최댓값 1.0으로 고정되고 심실 경계선이 뚜렷하게 복원됨을 검증 완료.

---

## 🗂️ 현재 원격 저장소 동기화 현황 (Git)
현재 모든 핵심 코드, 전처리 유틸리티, 테스트 스크립트 및 시각화용 이미지 자산은 대용량 파일(`.venv`, `data` 제외)을 제외하고 깔끔하게 원격 저장소에 반영되었습니다.
* **Repository**: [https://github.com/nodabnodab/CardioSeg.git](https://github.com/nodabnodab/CardioSeg.git)
* **커밋 히스토리**:
  1. `Initial commit: CardioSeg structure and interactive guide` (기본 기획서 & Streamlit 앱)
  2. `Add GPU verification script` (RTX 4070 Ti CUDA 검증 코드 추가)
  3. `Add preprocessing visualization script and comparison figure` (MONAI 전처리 비교 자산 추가)
  4. `Update app_guide with preprocessing comparison options` (대비 조절 토글이 반영된 웹 가이드 업데이트)
  5. `Implement patient-level 3-split and core preprocessing dataset loader` (3분할 분리 및 대비 필터 연동 데이터 로더 구현)

---

## 🚀 Next Steps (진행 예정 작업)
1. **데이터 증강 (Data Augmentation) 방식 협의 및 추가**:
   * 학습 오버피팅을 방지하기 위해 학습용 데이터셋에 무작위 회전, 스케일링, 패치 크롭 등의 데이터 증강(Augmentation) 규칙을 논의 후 추가.
2. **3D U-Net 신경망 설계 및 학습 스크립트 작성**:
   * RTX 4070 Ti 가속과 PyTorch AMP(Mixed Precision 16-bit)를 활용해 VRAM 용량 효율을 올리면서 오버랩 손실 함수(`DiceCELoss`)를 적용한 훈련 엔진 개발.
3. **임상 정량 수치 평가 모듈 연동**:
   * 검증 데이터셋에 대해 Dice Score 평가 후, 예측 마스크를 활용해 박출률(EF)과 질량(Mass) 오차율을 자동 역산하는 최종 비즈니스 로직 작성.
