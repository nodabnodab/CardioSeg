# 🫀 CardioSeg3D: 3D Cine-MRI Cardiac Segmentation & Clinical CDSS

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![MONAI](https://img.shields.io/badge/MONAI-1.2+-blueviolet.svg)](https://monai.io/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.25+-ff4b4b.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **CardioSeg3D**는 3D Cine-MRI 영상에서 심장의 핵심 3대 조직인 **우심실(RV), 심근(MYO), 좌심실(LV)**을 인공지능으로 자동 분할하고, 심실의 체적 변동률을 측정하여 **좌심실 박출률(LVEF)** 및 **심부전 위험도**를 실시간으로 판독하는 임상 의사결정 지원 시스템(CDSS) 프로토타입 플랫폼입니다.

---

## 🏥 플랫폼 구동 예시 (Clinical AI Dashboard)

이완기말(ED) 및 수축기말(ES) 프레임의 3차원 MRI 볼륨을 실시간 분할하여 LVEDV, LVESV, LVEF를 정량 계산하고, 의학적 소견 진단서와 3D 단면 마스크 오버레이를 렌더링합니다.

![AI 실시간 판독 및 진단서 발급 화면](assets/patient_081_ai_diagnosis.png)

---

## ✨ 핵심 기능 (Key Features)

1. **💡 심장 MRI 가이드 & 복셀 샌드박스**: 비전공의 대상 심장 해부학적 해설 및 Voxel Spacing 물리 부피 환산 계산기 탑재.
2. **⚙️ 3D 데이터 전처리 검증**: Spacing 정렬, 상하위 2% 클리핑 Contrast Windowing, 3D 표적 중심 크롭(`RandCropByPosNegLabeld`) 및 공간 증강 시각화.
3. **📈 학습 모니터링 & 대조**: Low-Res Baseline 모델 대비 High-Res 등방성 보간 모델의 훈련 수렴 추이 및 검증 메트릭 실시간 비교.
4. **🏥 실시간 AI 심부전 판독기 (CDSS)**: 
   * 환자별 MRI 데이터를 3D U-Net 추론 엔진 및 CCA(연결 성분 분석) 후처리로 자동 정화.
   * LVEF 산출 지표와 연동된 **심부전(HFrEF) 경보 및 임상 판독 소견서** 자동 작성.
   * 슬라이더를 통한 3D 단면 Z-Slice 스크롤 및 전문의 라벨(GT) vs AI 예측(Prediction) 시각 대조.

---

## 📊 성능 평가 및 임상 의의 (Results)

전체 검증/테스트 환자 20명에 대해 원본 임상 공간에서 3D U-Net 모델들을 종합 교차 검증한 결과입니다.

### 1. 세그멘테이션 다이렉트 정확도 (Test Dice Score)
* **High-Res + CCA 최종 모델**이 테스트셋 평균 Dice **84.01%**를 달성하며 Baseline 대비 **+1.99%** 향상되었습니다.
* 특히 치료 결정에 가장 중대한 영향을 미치는 **좌심실(LV) 성능은 90.11%**로 임상 적용 가능 수준(90% 선)을 돌파했습니다.

### 2. 임상 지표 오차 및 진단 일치율
* **LVEF 절대 평균 오차(MAE)**: 최종 모델 기준 **3.81%**
  * *의의*: 이는 심장 MRI 판독 전문의들의 **간 판독 편차(Inter-observer Variability)인 2.97% ~ 5.0%의 정중앙에 위치**하는 수치로, AI가 대학병원 전문의급 수준의 오차 정밀도로 LVEF를 정량 측정할 수 있음을 검증했습니다.
* **3단계 심부전 진단 일치율**: Baseline **80.0%** ➡️ 최종 모델 **85.0%**로 오진율 감소.

---

## 💻 빠른 시작 가이드 (Quick Start)

### 1. 가상환경 및 의존성 설치
본 프로젝트는 Windows 및 CUDA 환경에 최적화되어 있습니다.

```powershell
# 저장소 복제 및 이동
git clone https://github.com/nodabnodab/CardioSeg.git
cd CardioSeg

# 가상환경 구축 및 활성화
python -m venv .venv
.venv\Scripts\activate

# 패키지 설치
pip install -r requirements.txt
```

### 2. 마스터 대시보드 구동
```powershell
streamlit run app.py --server.port 8501
```
구동 후 브라우저에서 `http://localhost:8501`로 접속해 주세요.

### 3. 고해상도 3D U-Net 모델 학습 실행
```powershell
python train_hr.py
```

---

## 📂 디렉토리 구조 (Directory Structure)

* `app.py`: 의학 가이드, 전처리 검증, 학습 모니터 및 AI 판독기가 통합된 마스터 대시보드.
* `train_hr.py` / `train.py`: 고해상도(Z축 2.5mm) 및 기본 해상도(Z축 5.0mm) 3D U-Net 학습 엔진.
* `verify_clinical_impact.py`: 검증/테스트 환자 20명 대상 LVEF MAE 및 진단 일치율 자동 실측 도구.
* `evaluate_cca.py`: CCA 후처리 적용 전후 Dice 스코어 변동 정량 분석기.
* `src/`:
  * `model.py` / `model_hr.py`: Instance Normalization이 적용된 3D U-Net 아키텍처 정의.
  * `dataset.py`: MONAI 대비 Windowing 및 표적 크롭 데이터 파이프라인.
* `assets/`: 3D 증강 단면, 명암비 대조 및 학습 곡선 플롯 자산 폴더.
* `CardioSeg_Progress.md`: 기술 결정 사항, 성능 대조 및 글로벌 벤치마크 분석을 수록한 **상세 포트폴리오 리포트**.
