# 🫀 CardioSeg3D: 3D Cardiac MRI Multi-Structure Segmentation & Clinical Quantification Pipeline

이 프로젝트는 **NIfTI (.nii.gz)** 포맷의 3D Cine-MRI 데이터를 활용하여 심장의 핵심 구조인 **우심실(RV), 좌심실(LV), 심근(Myocardium)**을 3D로 분할(Segmentation)하고, 의학적 정량 지표인 **박출률(Ejection Fraction, EF)** 및 **심근 질량(Myocardial Mass)**을 자동 추출하는 End-to-End 의료 AI 파이프라인입니다. 

의료 영상 연구 및 임상 솔루션 분야에서 비전 AI 개발자에게 기대하는 핵심 기술 역량(3D 데이터 처리, 기하학적 메타데이터 파싱, 평가지표 검증)을 증명하도록 설계되었습니다.

---

## 1. 프로젝트 수행 배경 및 기존 문제점

### 📌 주제
* 3D 심장 MRI 다중 구조 분할 및 임상 지표 자동 정량화 (3D Cardiac MRI Segmentation & Quantification)

### ⚠️ 기존의 문제점
1. **수동 구획의 한계**: 심장 질환(심근경색, 심부전 등)의 정밀 진단을 위해서는 좌/우심실의 부피 및 심근 두께 측정이 필수적입니다. 그러나 현재 임상에서는 전문의가 환자의 3D MRI 슬라이스 수십 장을 넘겨가며 수작업으로 마스크를 그리는데, 이는 **20~30분 이상 소요**될 뿐만 아니라 **판독의의 주관적 편차**가 발생합니다.
2. **3D 공간적 일관성 부재**: 일반적인 2D CNN 기반 세그멘테이션 모델은 슬라이스 단면(2D Slice) 단위로만 학습하기 때문에 Z축 방향의 3D 공간적 연속성과 기하학적 구조를 인지하지 못합니다. 이로 인해 슬라이스 간 단절이 일어나며, 부피 역산 시 임상적으로 큰 오차를 초래합니다.
3. **의료 영상의 다양성(Heterogeneity)**: 스캐너 제조사나 촬영 장비에 따라 복셀 간격(Voxel Spacing)과 밝기 강도(Intensity)가 제각각이어서 단순 AI 모델은 도메인 변화에 매우 취약합니다.

---

## 2. 프로젝트 기대 효과

1. **판독 및 분석 시간 단축**: 전문의가 30분 이상 걸리던 심실 부피 측정 작업을 **수 초 이내로 자동화**하여 진단 업무 효율성을 극대화합니다.
2. **시뮬레이션 가교 역할**: 3D 공간상에서 기하학적으로 일관되고 닫힌(Closed) 3D 심장 볼륨 마스크를 획득하여, 3D 모델링 및 유체역학 시뮬레이션 등에 원활하게 연계할 수 있습니다.
3. **화이트박스(Explainable) 의료 AI**: 단순 오버랩 지표(Dice)만 제시하는 블랙박스 모델을 탈피하고, 의학적 지표(EF, 질량)를 정량 계산 및 오버레이 시각화하여 의료진이 신뢰할 수 있는 임상 정보를 함께 제공합니다.

---

## 3. 데이터 정보 및 정확한 출처

* **데이터셋 이름**: **ACDC (Automated Cardiac Diagnosis Challenge) Dataset** (MICCAI 2017 Challenge)
* **공식 출처**: 프랑스 CREATIS 연구소 주관 ACDC 챌린지 ([공식 웹사이트](https://www.creatis.insa-lyon.fr/Challenge/acdc/))
* **포맷**: **NIfTI (.nii.gz)** - 3D 영상 데이터 및 전문가가 라벨링한 Ground Truth 마스크 제공.
* **데이터 구성**:
  - 총 150명의 환자 데이터 (학습용 100 케이스, 테스트용 50 케이스).
  - 심장의 수축과 이완 주기를 나타내는 Cine-MRI 시계열 볼륨 중에서 **내막기(End-Diastole, ED)**와 **외막기(End-Systole, ES)**의 정지 프레임(3D Volume)에 대한 라벨 제공.
  - **라벨 구조**: 0 (Background), 1 (Right Ventricle Cavity - RV), 2 (Myocardium - MYO), 3 (Left Ventricle Cavity - LV).

---

## 4. 사용 기술 및 하드웨어 환경

### 💻 하드웨어 환경 (Hardware Setup)
* **GPU**: **NVIDIA GeForce RTX 4070 Ti (12GB VRAM)**
* **가속 기술**: CUDA Cores 가속, PyTorch AMP (Automatic Mixed Precision)를 활용한 16-bit 부동소수점 연산으로 VRAM 절약 및 학습 속도 극대화.

### 🛠️ 소프트웨어 기술 스택 (Software Tech Stack)
* **Deep Learning Framework**: PyTorch
* **Medical Image Library**: **MONAI (Medical Open Network for AI)**
  - 3D Spacing Resampling, 3D Data Augmentation, 3D U-Net 네트워크 설계에 활용.
* **Image File Format Parsing**: SimpleITK, Nibabel (NIfTI 헤더 데이터 파싱 및 복셀 부피 역산에 사용).
* **Evaluation & Post-processing**: Scikit-image (Connected Component Analysis), MedPy
* **Interactive Demo UI**: Streamlit (3D 슬라이스 스크롤 뷰어 및 임상 계산 지표 시각화 대시보드).

---

## 5. 베이스라인 모델 설정 및 극복 증명 방안

### 📉 Baseline 정의
* **모델 및 학습 환경**: 기본적인 3D U-Net 모델을 사용하되, 의료 특화 전처리(물리적 간격 보정, Z-score 밝기 정규화 등) 없이 단순히 Image Resize만 적용하고 표준 Cross-Entropy Loss로 학습한 모델.
* **베이스라인 한계**: 각 MRI 파일의 Voxel Spacing이 보정되지 않아 해상도가 뭉개지며, 좌/우심실 면적 대비 배경이 너무 넓어 클래스 불균형에 의해 심실 경계선 검출이 무너집니다.

### 🚀 본 프로젝트의 개선점 (개선 방안)
1. **기하학적 왜곡 방지 (Spacing Resampling)**: `Spacingd` 트랜스폼을 적용하여 서로 다른 복셀 물리 공간을 `[1.25, 1.25, 5.0] mm`로 통일해 3D 공간 뒤틀림을 해결합니다.
2. **클래스 불균형 해소 (DiceCELoss)**: 배경 영역 가중치를 낮추고 오버랩과 픽셀 오차를 동시에 학습하는 `DiceCELoss`를 채택합니다.
3. **가양성 제거 후처리 (Connected Component Analysis)**: 모델 예측 출력물에 연결 성분 분석을 적용해 주 심장 구조물 이외의 곳에 흩뿌려진 미세 노이즈 픽셀(False Positives)을 깨끗이 지웁니다.

### 📊 성능 증명 방식 (How to Prove)
본 프로젝트가 베이스라인보다 우수함을 다음 3가지 핵심 평가지표를 통해 객관적으로 비교 검증합니다.

1. **Dice Similarity Coefficient (DSC) 비교**:
   - 3D 볼륨 내 예측값과 정답지의 영역 일치도를 비교 평가합니다. (목표치: 평균 DSC **88% 이상** 달성 및 베이스라인 대비 3% 이상 우위 확보).
2. **Hausdorff Distance 95 (HD95) 비교**:
   - 경계면 예측 오차(mm 단위)를 측정하여 경계선 획정 정밀도를 비교합니다. (목표치: HD95 수치가 베이스라인보다 현저히 감소하여 경계선이 선명해짐을 증명).
3. **임상 지표 오차 평가 (Clinical Error)**:
   - 예측 마스크를 활용해 계산된 **좌심실 박출률 (Ejection Fraction, EF)** 지표가 의사가 레이블링한 정답지 마스크로 계산된 진짜 EF 지표와 비교했을 때 평균 오차(MAE)가 몇 % 이내로 떨어지는지 검증합니다. 베이스라인 모델의 예측 결과보다 실제 임상값과의 정량 오차가 확연히 낮아짐을 증명합니다.

---

## 📅 1-Week 상세 타임라인 & 태스크

### 🛑 1일차: 데이터셋 확보 및 NIfTI 구조 분석
- [ ] ACDC NIfTI 데이터셋 로컬 환경에 다운로드 및 구조 설정 (`data/ACDC`)
- [ ] `nibabel`을 사용하여 NIfTI 이미지의 affine matrix 및 shape, spacing 정보 로그 분석

### 🔍 2일차: MONAI 기반 3D 전처리 파이프라인 구축
- [ ] `Spacingd`를 사용한 isotropic/anisotropic 복셀 간격 보정 적용
- [ ] `CropForegroundd` 및 밝기 편차 해소를 위한 `NormalizeIntensityd` 적용 후 전처리 비포/애프터 비교 이미지 생성
- [ ] 학습 효율을 위해 3D patch crop 적용 데이터 로더 완성

### 🤖 3일차: 3D U-Net 및 SegResNet 구조 설계
- [ ] RTX 4070 Ti의 VRAM 환경에 최적화된 3D U-Net/SegResNet 네트워크 정의
- [ ] `torch.cuda.amp.autocast`를 활용한 Mixed Precision 학습 파이프라인 연동

### 🔥 4~5일차: DiceCELoss 기반 고속 훈련 및 검증
- [ ] `DiceCELoss` 결합 손실 함수 설계 및 훈련 모니터링
- [ ] Validation Loss/Dice Score 기반 최적의 모델 가중치 체크포인트 저장

### 📈 6일차: 연결 성분 분석(CCA) 및 임상 정량 계산 구현
- [ ] `KeepLargestConnectedComponentd` 필터를 적용한 예측 마스크 미세 노이즈 제거
- [ ] ED/ES 프레임 간의 LV 부피 변화를 감지하여 박출률(EF) 및 심근 질량을 계산하는 수학적 모듈 개발

### 💻 7일차: Streamlit 3D Slice Viewer 구현 및 성과 문서화
- [ ] Streamlit 웹 화면에 NIfTI 드래그앤드롭 및 슬라이더 스크롤 시각화 기능 구축
- [ ] 베이스라인 대비 향상된 성능(Dice, HD95, EF 오차) 표 및 오버레이 contour 비교 자료 정리
