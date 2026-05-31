import os
import streamlit as st
from PIL import Image

# Set Page Config for Dev Explorer
st.set_page_config(
    page_title="CardioSeg3D: Developer Explorer",
    page_icon="⚙️",
    layout="wide"
)

# Styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.5rem;
        font-weight: 800;
        color: #1f77b4;
        margin-bottom: 0.5rem;
    }
    .sub-title {
        font-size: 1.1rem;
        color: #555;
        margin-bottom: 2rem;
    }
    .stat-card {
        background-color: #f7f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #1f77b4;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Define Paths
base_dir = os.path.dirname(os.path.abspath(__file__))
assets_dir = os.path.join(base_dir, "assets")

st.markdown('<h1 class="main-title">⚙️ CardioSeg3D: Developer Data Explorer</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">학습용 3D 전처리(Preprocessing) 및 실시간 데이터 증강(Augmentation) 파이프라인 시각 검증 패널</p>', unsafe_allow_html=True)

# Sidebar
st.sidebar.title("🛠️ Developer Panel")
st.sidebar.markdown("""
**데이터 파이프라인 관리 모듈**
***
본 패널은 데이터 전처리 강도 범위 및 실시간 크롭 패치들의 기하학적/명암 변동 상태를 모니터링하기 위해 구성되었습니다.
""")

st.sidebar.info("""
📊 **학습 데이터 분할 현황 (ACDC)**
- **Train (학습)**: 80명 (160 볼륨)
- **Validation (검증)**: 10명 (20 볼륨)
- **Test (테스트)**: 10명 (20 볼륨)
""")

# Tabs for Developer
tab_prep, tab_aug, tab_train = st.tabs([
    "⚙️ 3D 데이터 전처리 (Preprocessing)",
    "🔄 3D 데이터 증강 & 크롭 (Augmentation)",
    "📈 학습 곡선 모니터링 (Training Monitor)"
])

# Tab 1: Preprocessing
with tab_prep:
    st.subheader("1. 3D Spacing & Contrast Windowing 전처리 결과")
    st.write("""
    모델 학습에 투입되기 전 해상도 정렬(`1.25 x 1.25 x 5.0 mm`) 및 전경 추출을 거치고, 
    Z-score 정규화에 의료용 윈도잉(상하위 2% 클리핑)을 씌워 선명도를 복원한 비포/애프터 비교입니다.
    """)
    
    prep_option = st.radio(
        "시각화 모드 선택:",
        ["기본 전처리 (전체 범위 시각화 - Washed-out 대비)", "대비 개선 전처리 (의료용 Contrast Windowing 적용 - 선명함)"],
        horizontal=True
    )
    
    img_prep_path = os.path.join(assets_dir, "preprocessing_comparison.png")
    img_prep_contrast_path = os.path.join(assets_dir, "preprocessing_comparison_contrast.png")
    
    if prep_option == "기본 전처리 (전체 범위 시각화 - Washed-out 대비)":
        if os.path.exists(img_prep_path):
            st.image(Image.open(img_prep_path), caption="기본 전처리 결과 (2x2)", use_container_width=True)
            st.info("💡 **특징:** 밝기 정규화 후 단순히 최소/최대값 범위로 그려 대비가 분산되어 약간 흐려 보입니다.")
        else:
            st.warning("전처리 비교 이미지가 존재하지 않습니다. `visualize_preprocessing.py`를 실행해 주세요.")
    else:
        if os.path.exists(img_prep_contrast_path):
            st.image(Image.open(img_prep_contrast_path), caption="대비 개선(Contrast Windowing) 적용 결과 (2x2)", use_container_width=True)
            st.success("✨ **특징:** 밝기 노이즈(상위/하위 2%)를 깎아내고 중심부 장기 대비만 극대화하여 경계선과 픽셀들이 선명하게 복원되었습니다.")
        else:
            st.warning("대비 개선 이미지가 존재하지 않습니다. `visualize_tuning.py`를 실행해 주세요.")

# Tab 2: Augmentation
with tab_aug:
    st.subheader("2. 원본 데이터셋 대비 실시간 3D 데이터 증강 및 패치 크롭 검출 비교")
    st.write("""
    학습 볼륨으로부터 어떻게 실시간으로 회전/스케일링이 가해지고, 
    심장 구조체 근처에서 `128 x 128 x 8` 크기의 3D 패치 4개가 추출되었는지를 **원본 3D 볼륨(좌측)**과 **4개의 증강 패치(우측)**를 매핑하여 시각화한 결과입니다.
    """)
    
    # Let the developer select the case to study
    case_option = st.selectbox(
        "🔎 세부 검증용 환자 사례(Case) 선택:",
        ["Case 1 (Patient 001 - 정상 박출률 심장)", "Case 2 (Patient 006 - 중간 크기 심장)", "Case 3 (Patient 011 - 우심실 확장 양상 심장)"]
    )
    
    case_to_file = {
        "Case 1 (Patient 001 - 정상 박출률 심장)": "aug_case_1.png",
        "Case 2 (Patient 006 - 중간 크기 심장)": "aug_case_2.png",
        "Case 3 (Patient 011 - 우심실 확장 양상 심장)": "aug_case_3.png"
    }
    
    img_name = case_to_file[case_option]
    img_path = os.path.join(assets_dir, img_name)
    
    if os.path.exists(img_path):
        st.image(Image.open(img_path), caption=f"실시간 3D 데이터 증강 및 4개 패스 비교 - {case_option}", use_container_width=True)
        st.success(f"✨ **{case_option} 분석 완료:**")
        st.info("""
        💡 **개발자 확인 포인트:**
        - **원본 대조 (좌측 2개 이미지)**: Spacing 해상도가 보정되고 대비가 윈도잉 정규화된 3차원 원본 볼륨의 중간 단면입니다.
        - **증강 패치 대조 (우측 8개 이미지)**: 원본으로부터 Z축 기준 회전(Rotation) 및 스케일링(Scale Shift)이 실시간으로 적용되고, 심실 경계(RV/LV/MYO)를 표적으로 성공적으로 잘라낸 모습입니다.
        - **클래스 균형(Class Balancing)**: 각 패치에 1번(우심실), 2번(심근), 3번(좌심실) 중 최소 1개 이상의 타겟 조직이 반드시 균일하게 함유되도록 크롭하여 학습 수렴 속도가 매우 빠릅니다.
        """)
    else:
        st.warning(f"선택한 이미지({img_name})가 존재하지 않습니다. `visualize_aug_cases.py`를 먼저 작동해 주세요.")

# Tab 3: Training Monitor
with tab_train:
    st.subheader("3. 3D U-Net 실시간 학습 곡선 모니터링")
    st.write("""
    터미널에서 `python train.py`를 실행하여 훈련이 활성화되면, 실시간으로 에포크별 오차(Loss) 감소 추이와 
    검증(Validation)용 3차원 슬라이딩 윈도우 Dice 성능 곡선이 기록되어 아래 그래프에 업데이트됩니다.
    """)
    
    import json
    history_file = os.path.join(assets_dir, "training_history.json")
    plot_file = os.path.join(assets_dir, "training_curves.png")
    
    if os.path.exists(history_file) and os.path.exists(plot_file):
        try:
            with open(history_file, "r") as f:
                history = json.load(f)
        except Exception:
            history = {}
            
        epochs = history.get("epoch", [])
        train_loss = history.get("train_loss", [])
        val_mean_dice = history.get("val_mean_dice", [])
        
        if epochs:
            latest_epoch = epochs[-1]
            latest_loss = train_loss[-1]
            
            # Find best validation score (ignoring placeholder zeros)
            valid_dices = [d for d in val_mean_dice if d > 0]
            best_dice = max(valid_dices) if valid_dices else 0.0
            
            best_epoch = "-"
            if best_dice > 0:
                best_idx = val_mean_dice.index(best_dice)
                best_epoch = epochs[best_idx]
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("현재 진행 에포크", f"{latest_epoch} / 150")
            with col2:
                st.metric("최우수 평균 Dice Score", f"{best_dice:.4f}" if best_dice > 0 else "측정 전", help="배경을 제외한 우심실, 심근, 좌심실의 3D 평균 정확도 (Best Epoch: {best_epoch})")
            with col3:
                st.metric("현재 훈련 Loss", f"{latest_loss:.4f}")
                
            st.image(Image.open(plot_file), caption="실시간 3D U-Net 훈련 추이 그래프 (Loss & Dice)", use_container_width=True)
            
            # Show history log summary
            with st.expander("📋 자세한 에포크별 성능 로그 확인"):
                st.dataframe(history, use_container_width=True)
        else:
            st.info("학습 기록 파일은 존재하지만, 아직 에포크가 시작되지 않았습니다.")
    else:
        st.info("💡 **실시간 학습 모니터 작동 방법:**")
        st.warning("현재 학습 히스토리 파일이 존재하지 않습니다. 먼저 터미널에서 학습을 구동해 주세요.")
        st.markdown("""
        1. 터미널을 열고 가상환경을 활성화합니다:
           ```powershell
           .venv\\Scripts\\activate
           ```
        2. 훈련 명령어를 실행하여 3D U-Net 학습을 구동시킵니다:
           ```powershell
           python train.py
           ```
        3. 학습이 활성화되고 1 에포크가 끝나는 즉시, 본 화면에 실시간 손실 곡선 및 정확도 메트릭 패널이 자동 로딩됩니다!
        """)
