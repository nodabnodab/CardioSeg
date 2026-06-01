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
    기본 저해상도(Low-Res) 모델과 고해상도 업스케일(High-Res) 모델의 학습 진척률 및 평가지표를 비교/모니터링합니다.
    """)
    
    sub_tab_lr, sub_tab_hr, sub_tab_comp = st.tabs([
        "📉 기본 모델 (Low-Res)", 
        "🚀 업스케일 모델 (High-Res)", 
        "📊 두 모델 성능 비교"
    ])
    
    import json
    
    # 3.1. Low-Res Tab
    with sub_tab_lr:
        history_file_lr = os.path.join(assets_dir, "training_history.json")
        plot_file_lr = os.path.join(assets_dir, "training_curves.png")
        
        if os.path.exists(history_file_lr) and os.path.exists(plot_file_lr):
            try:
                with open(history_file_lr, "r") as f:
                    history_lr = json.load(f)
            except Exception:
                history_lr = {}
                
            epochs_lr = history_lr.get("epoch", [])
            train_loss_lr = history_lr.get("train_loss", [])
            val_mean_dice_lr = history_lr.get("val_mean_dice", [])
            
            if epochs_lr:
                latest_epoch_lr = epochs_lr[-1]
                latest_loss_lr = train_loss_lr[-1]
                valid_dices_lr = [d for d in val_mean_dice_lr if d > 0]
                best_dice_lr = max(valid_dices_lr) if valid_dices_lr else 0.0
                
                best_epoch_lr = "-"
                if best_dice_lr > 0:
                    best_idx_lr = val_mean_dice_lr.index(best_dice_lr)
                    best_epoch_lr = epochs_lr[best_idx_lr]
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("현재 진행 에포크", f"{latest_epoch_lr} / 150")
                with col2:
                    st.metric("최우수 평균 Dice Score", f"{best_dice_lr:.4f}" if best_dice_lr > 0 else "측정 전", help=f"Best Epoch: {best_epoch_lr}")
                with col3:
                    st.metric("현재 훈련 Loss", f"{latest_loss_lr:.4f}")
                    
                st.image(Image.open(plot_file_lr), caption="실시간 Low-Res 훈련 추이 그래프 (Loss & Dice)", use_container_width=True)
                with st.expander("📋 자세한 에포크별 성능 로그 확인"):
                    st.dataframe(history_lr, use_container_width=True)
            else:
                st.info("학습 기록 파일은 존재하지만, 아직 에포크가 시작되지 않았습니다.")
        else:
            st.info("💡 **실시간 학습 모니터 작동 방법 (Low-Res):**")
            st.warning("현재 기본 모델의 학습 히스토리 파일이 존재하지 않습니다.")
            st.markdown("""
            1. 터미널을 열고 가상환경을 활성화합니다:
               ```powershell
               .venv\\Scripts\\activate
               ```
            2. 훈련 명령어를 실행하여 기본 모델 학습을 구동시킵니다:
               ```powershell
               python train.py
               ```
            """)
            
    # 3.2. High-Res Tab
    with sub_tab_hr:
        history_file_hr = os.path.join(assets_dir, "training_history_hr.json")
        plot_file_hr = os.path.join(assets_dir, "training_curves_hr.png")
        
        if os.path.exists(history_file_hr) and os.path.exists(plot_file_hr):
            try:
                with open(history_file_hr, "r") as f:
                    history_hr = json.load(f)
            except Exception:
                history_hr = {}
                
            epochs_hr = history_hr.get("epoch", [])
            train_loss_hr = history_hr.get("train_loss", [])
            val_mean_dice_hr = history_hr.get("val_mean_dice", [])
            
            if epochs_hr:
                latest_epoch_hr = epochs_hr[-1]
                latest_loss_hr = train_loss_hr[-1]
                valid_dices_hr = [d for d in val_mean_dice_hr if d > 0]
                best_dice_hr = max(valid_dices_hr) if valid_dices_hr else 0.0
                
                best_epoch_hr = "-"
                if best_dice_hr > 0:
                    best_idx_hr = val_mean_dice_hr.index(best_dice_hr)
                    best_epoch_hr = epochs_hr[best_idx_hr]
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("현재 진행 에포크", f"{latest_epoch_hr} / 150")
                with col2:
                    st.metric("최우수 평균 Dice Score", f"{best_dice_hr:.4f}" if best_dice_hr > 0 else "측정 전", help=f"Best Epoch: {best_epoch_hr}")
                with col3:
                    st.metric("현재 훈련 Loss", f"{latest_loss_hr:.4f}")
                    
                st.image(Image.open(plot_file_hr), caption="실시간 High-Res 훈련 추이 그래프 (Loss & Dice)", use_container_width=True)
                with st.expander("📋 자세한 에포크별 성능 로그 확인"):
                    st.dataframe(history_hr, use_container_width=True)
            else:
                st.info("학습 기록 파일은 존재하지만, 아직 에포크가 시작되지 않았습니다.")
        else:
            st.info("💡 **실시간 학습 모니터 작동 방법 (High-Res):**")
            st.warning("현재 고해상도 모델의 학습 히스토리 파일이 존재하지 않습니다.")
            st.markdown("""
            1. 터미널을 열고 가상환경을 활성화합니다:
               ```powershell
               .venv\\Scripts\\activate
               ```
            2. 훈련 명령어를 실행하여 고해상도 업스케일 모델 학습을 구동시킵니다:
               ```powershell
               python train_hr.py
               ```
            """)

    # 3.3. Comparison Tab
    with sub_tab_comp:
        st.write("### 📊 기본 모델 (Low-Res) vs 업스케일 모델 (High-Res) 비교")
        
        history_file_lr = os.path.join(assets_dir, "training_history.json")
        history_file_hr = os.path.join(assets_dir, "training_history_hr.json")
        
        lr_loaded = False
        hr_loaded = False
        
        if os.path.exists(history_file_lr):
            try:
                with open(history_file_lr, "r") as f:
                    hist_lr = json.load(f)
                lr_loaded = len(hist_lr.get("epoch", [])) > 0
            except Exception:
                pass
                
        if os.path.exists(history_file_hr):
            try:
                with open(history_file_hr, "r") as f:
                    hist_hr = json.load(f)
                hr_loaded = len(hist_hr.get("epoch", [])) > 0
            except Exception:
                pass
                
        if lr_loaded or hr_loaded:
            # Prepare comparison table
            lr_best_dice = 0.0
            lr_rv = 0.0; lr_myo = 0.0; lr_lv = 0.0
            lr_test_mean = 0.0; lr_test_rv = 0.0; lr_test_myo = 0.0; lr_test_lv = 0.0
            if lr_loaded:
                val_mean_lr = hist_lr.get("val_mean_dice", [])
                valid_lr = [d for d in val_mean_lr if d > 0]
                if valid_lr:
                    lr_best_dice = max(valid_lr)
                    best_idx = val_mean_lr.index(lr_best_dice)
                    lr_rv = hist_lr.get("val_rv_dice", [])[best_idx]
                    lr_myo = hist_lr.get("val_myo_dice", [])[best_idx]
                    lr_lv = hist_lr.get("val_lv_dice", [])[best_idx]
                    
                    if "test_mean_dice" in hist_lr and len(hist_lr["test_mean_dice"]) > best_idx:
                        lr_test_mean = hist_lr["test_mean_dice"][best_idx]
                        lr_test_rv = hist_lr["test_rv_dice"][best_idx]
                        lr_test_myo = hist_lr["test_myo_dice"][best_idx]
                        lr_test_lv = hist_lr["test_lv_dice"][best_idx]
            
            hr_best_dice = 0.0
            hr_rv = 0.0; hr_myo = 0.0; hr_lv = 0.0
            hr_test_mean = 0.0; hr_test_rv = 0.0; hr_test_myo = 0.0; hr_test_lv = 0.0
            if hr_loaded:
                val_mean_hr = hist_hr.get("val_mean_dice", [])
                valid_hr = [d for d in val_mean_hr if d > 0]
                if valid_hr:
                    hr_best_dice = max(valid_hr)
                    best_idx = val_mean_hr.index(hr_best_dice)
                    hr_rv = hist_hr.get("val_rv_dice", [])[best_idx]
                    hr_myo = hist_hr.get("val_myo_dice", [])[best_idx]
                    hr_lv = hist_hr.get("val_lv_dice", [])[best_idx]
                    
                    if "test_mean_dice" in hist_hr and len(hist_hr["test_mean_dice"]) > best_idx:
                        hr_test_mean = hist_hr["test_mean_dice"][best_idx]
                        hr_test_rv = hist_hr["test_rv_dice"][best_idx]
                        hr_test_myo = hist_hr["test_myo_dice"][best_idx]
                        hr_test_lv = hist_hr["test_lv_dice"][best_idx]
            
            def get_delta_str(v_hr, v_lr):
                if v_lr == 0.0 or v_hr == 0.0:
                    return "-"
                diff = v_hr - v_lr
                color = "🟢 +" if diff >= 0 else "🔴 "
                return f"{color}{diff:.4f}"
            
            st.markdown(f"""
            | 평가지표 (Dice) | 기본 모델 (Low-Res) | 업스케일 모델 (High-Res) | 성능 차이 (Delta) |
            |---|---|---|---|
            | **[검증] 평균 Dice Score** | {f"{lr_best_dice:.4f}" if lr_best_dice > 0 else "-"} | {f"{hr_best_dice:.4f}" if hr_best_dice > 0 else "-"} | **{get_delta_str(hr_best_dice, lr_best_dice)}** |
            | [검증] 우심실 (RV Dice) | {f"{lr_rv:.4f}" if lr_rv > 0 else "-"} | {f"{hr_rv:.4f}" if hr_rv > 0 else "-"} | {get_delta_str(hr_rv, lr_rv)} |
            | [검증] 심근 (MYO Dice) | {f"{lr_myo:.4f}" if lr_myo > 0 else "-"} | {f"{hr_myo:.4f}" if hr_myo > 0 else "-"} | {get_delta_str(hr_myo, lr_myo)} |
            | [검증] 좌심실 (LV Dice) | {f"{lr_lv:.4f}" if lr_lv > 0 else "-"} | {f"{hr_lv:.4f}" if hr_lv > 0 else "-"} | {get_delta_str(hr_lv, lr_lv)} |
            |---|---|---|---|
            | **[테스트] 평균 Dice Score** | {f"{lr_test_mean:.4f}" if lr_test_mean > 0 else "-"} | {f"{hr_test_mean:.4f}" if hr_test_mean > 0 else "-"} | **{get_delta_str(hr_test_mean, lr_test_mean)}** |
            | [테스트] 우심실 (RV Dice) | {f"{lr_test_rv:.4f}" if lr_test_rv > 0 else "-"} | {f"{hr_test_rv:.4f}" if hr_test_rv > 0 else "-"} | {get_delta_str(hr_test_rv, lr_test_rv)} |
            | [테스트] 심근 (MYO Dice) | {f"{lr_test_myo:.4f}" if lr_test_myo > 0 else "-"} | {f"{hr_test_myo:.4f}" if hr_test_myo > 0 else "-"} | {get_delta_str(hr_test_myo, lr_test_myo)} |
            | [테스트] 좌심실 (LV Dice) | {f"{lr_test_lv:.4f}" if lr_test_lv > 0 else "-"} | {f"{hr_test_lv:.4f}" if hr_test_lv > 0 else "-"} | {get_delta_str(hr_test_lv, lr_test_lv)} |
            """, unsafe_allow_html=True)
            
            if lr_loaded and hr_loaded:
                diff_mean = hr_best_dice - lr_best_dice
                
                st.write("### 💡 성능 실험 요약 및 임상적 해석")
                if diff_mean > 0:
                    st.success(f"🎉 **고해상도 업스케일 모델이 평균 Dice에서 {diff_mean:.4f}만큼 향상되었습니다!**")
                else:
                    st.info("두 모델이 학습 진행 중입니다. 학습이 완료되면 최종 성능 비교 수치가 확정됩니다.")
                    
                st.markdown(f"""
                - **심근(Myocardium) 정확도 변동 ({get_delta_str(hr_myo, lr_myo)})**:
                  - 심근은 두께가 얇아 저해상도 Z-axis 슬라이스 간격(5.0mm)에서는 부분 체적 효과(Partial Volume Effect)의 영향을 가장 크게 받습니다.
                  - Z-axis 해상도를 2.5mm로 업스케일링함에 따라 복셀 크기가 촘촘해져 심근의 미세한 경계면을 훨씬 정밀하게 탐지하는 효과가 나타납니다.
                - **3D 공간적 일관성**:
                  - 고해상도 모델은 패치 크기를 `128x128x16`로 2배 확대하고 strides를 `(2,2,2)` 등방성 구조로 대칭화하여 Z축의 불연속적인 슬라이딩 윈도우 끊김 현상을 물리적으로 개선했습니다.
                """)
            else:
                st.info("💡 **비교 뷰 안내**: 두 모델의 학습 이력이 모두 생성되면, 세부 메트릭 차이 및 물리 보간 분석 보고서가 여기에 자동으로 채워집니다!")
        else:
            st.info("아직 학습된 이력이 존재하지 않습니다. 좌측의 두 탭에서 모델 학습 가이드를 참조해 주세요.")
