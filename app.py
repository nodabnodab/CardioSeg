import os
import re
import json
import torch
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from PIL import Image
import nibabel as nib

from monai.inferers import sliding_window_inference
from monai.transforms import AsDiscrete, Spacing

from src.dataset import get_acdc_splits, get_val_test_transforms
from src.model import get_3d_unet
from src.model_hr import get_3d_unet_hr

# Set Page Config for Master Dashboard
st.set_page_config(
    page_title="CardioSeg3D: Master Dashboard",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling (Light/Dark Mode Adaptive, Premium Feel)
st.markdown("""
<style>
    .main-title {
        font-size: 2.8rem;
        font-weight: 800;
        color: #ff4b4b;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.1rem;
        color: #555;
        margin-bottom: 2rem;
    }
    .metric-box {
        background-color: var(--secondary-background-color);
        color: var(--text-color);
        padding: 1.2rem;
        border-radius: 10px;
        border-left: 5px solid #ff4b4b;
        margin-bottom: 1rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
    }
    .calc-box {
        background-color: rgba(31, 119, 180, 0.08);
        color: var(--text-color);
        padding: 1.2rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        margin-bottom: 1rem;
        border: 1px solid rgba(31, 119, 180, 0.15);
    }
    .clinical-card {
        background-color: #f7f9fa;
        color: #333;
        padding: 1.5rem;
        border-radius: 10px;
        border: 1px solid #e1e4e6;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Define Paths
base_dir = os.path.dirname(os.path.abspath(__file__))
assets_dir = os.path.join(base_dir, "assets")

# Images
img_raw_path = os.path.join(assets_dir, "mri_raw.png")
img_mask_path = os.path.join(assets_dir, "mask_only.png")
img_overlay_path = os.path.join(assets_dir, "mri_overlay.png")

# Cache Models
@st.cache_resource
def load_baseline_model(device):
    model = get_3d_unet(1, 4).to(device)
    lr_weights = os.path.join(base_dir, "best_metric_model.pth")
    if os.path.exists(lr_weights):
        model.load_state_dict(torch.load(lr_weights, map_location=device))
    model.eval()
    return model

@st.cache_resource
def load_hr_model(device):
    model = get_3d_unet_hr(1, 4).to(device)
    hr_weights = os.path.join(base_dir, "best_metric_model_hr.pth")
    if os.path.exists(hr_weights):
        model.load_state_dict(torch.load(hr_weights, map_location=device))
    model.eval()
    return model

# Sidebar Configuration
st.sidebar.title("🫀 CardioSeg3D Master")
st.sidebar.markdown("""
**심장 MRI 다중 구조 분할 및 임상 지표 자동 정량화 플랫폼**
***
본 플랫폼은 3D Cine-MRI 데이터를 전처리하고 학습 곡선을 분석할 뿐 아니라, 진단 현장에서 실시간으로 심부전을 판독하는 의료 AI 데모를 탑재하고 있습니다.
""")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
st.sidebar.success(f"📟 **연산 가속 장치**: {str(device).upper()}")
st.sidebar.info("""
📊 **데이터셋 구성 (ACDC)**
- 학습(Train): 80명 (160 볼륨)
- 검증(Val): 10명 (20 볼륨)
- 테스트(Test): 10명 (20 볼륨)
""")

# Title Banner
st.markdown('<h1 class="main-title">🫀 CardioSeg3D: 종합 의료 AI 플랫폼</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Cine-MRI 기반 심실 다중 구조 분할 및 AI 실시간 심부전 진단 보조 시스템</p>', unsafe_allow_html=True)

# Master Tabs
tab_guide, tab_pipe, tab_monitor, tab_clinical = st.tabs([
    "💡 1. 임상 가이드 & 기초 교육",
    "⚙️ 2. 데이터 파이프라인 검증",
    "📈 3. 학습 곡선 & 모델 비교",
    "🏥 4. 실시간 AI 심부전 판독기"
])

# ==============================================================================
# TAB 1: Clinical Guide & Sandbox
# ==============================================================================
with tab_guide:
    st.subheader("심장의 기본 구조와 3대 핵심 타겟")
    st.write("의료 인공지능(AI)은 심장 MRI 영상을 분석해 아래의 3가지 장기 영역을 자동으로 분할합니다.")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div class="metric-box">
            <h4>🔵 우심실 (Right Ventricle - RV)</h4>
            <p><b>라벨: 1번 (Blue)</b></p>
            <p>전신을 돌고 산소가 고갈되어 돌아온 혈액을 받아 폐로 보내 산소를 충전시키는 펌프 역할을 수행합니다.</p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="metric-box" style="border-left-color: #ff7f0e;">
            <h4>🟠 심근 (Myocardium - MYO)</h4>
            <p><b>라벨: 2번 (Orange)</b></p>
            <p>좌심실 혈액 주머니를 둘러싼 두꺼운 심장 근육 벽으로, 수축과 이완의 물리 운동을 직접 일으킵니다.</p>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="metric-box" style="border-left-color: #d62728;">
            <h4>🔴 좌심실 (Left Ventricle - LV)</h4>
            <p><b>라벨: 3번 (Red)</b></p>
            <p>폐에서 산소를 공급받은 혈액을 강하게 수축하여 대동맥을 통해 전신으로 전달하는 핵심 펌프입니다.</p>
        </div>
        """, unsafe_allow_html=True)
        
    st.write("---")
    st.subheader("3D 데이터셋 해상도 시각화")
    col_img1, col_img2, col_img3 = st.columns(3)
    if os.path.exists(img_raw_path) and os.path.exists(img_mask_path) and os.path.exists(img_overlay_path):
        col_img1.image(Image.open(img_raw_path), caption="1. 원본 심장 MRI 단면", use_container_width=True)
        col_img2.image(Image.open(img_mask_path), caption="2. 전문의의 라벨링 마스크", use_container_width=True)
        col_img3.image(Image.open(img_overlay_path), caption="3. MRI 레이블 오버레이 결과", use_container_width=True)
    else:
        st.warning("안내: `assets` 폴더 내에 시각화 자산 이미지가 존재하지 않습니다.")
        
    st.write("---")
    st.subheader("🧮 복셀 물리 부피 계산 샌드박스")
    calc_col1, calc_col2 = st.columns([1, 1])
    with calc_col1:
        dx = st.slider("가로 복셀 크기 (dx, mm)", min_value=0.5, max_value=3.0, value=1.5625, step=0.1)
        dy = st.slider("세로 복셀 크기 (dy, mm)", min_value=0.5, max_value=3.0, value=1.5625, step=0.1)
        dz = st.slider("Z축 슬라이스 두께 (dz, mm)", min_value=1.0, max_value=15.0, value=10.0, step=0.5)
        voxel_count = st.number_input("검출된 좌심실(LV) 복셀 수 (개)", min_value=100, max_value=50000, value=5000, step=100)
    with calc_col2:
        voxel_vol_mm3 = dx * dy * dz
        total_vol_mm3 = voxel_count * voxel_vol_mm3
        total_vol_ml = total_vol_mm3 / 1000.0
        
        st.markdown(f"""
        <div class="calc-box">
            <h4>📊 복셀 부피 계산 결과</h4>
            <hr>
            <p>• <b>단일 복셀 부피:</b> {voxel_vol_mm3:.4f} mm³</p>
            <p>• <b>총 체적 (mm³):</b> {total_vol_mm3:.2f} mm³</p>
            <p>• <b>최종 물리적 부피 (mL):</b> <b>{total_vol_ml:.2f} mL</b></p>
            <small>※ 의료 영상인 NIfTI는 Voxel Spacing 메타데이터가 기록되어 있어 3차원 해상도를 부피(mL) 단위로 정확히 환산 가능합니다.</small>
        </div>
        """, unsafe_allow_html=True)

# ==============================================================================
# TAB 2: Data Pipeline Verification
# ==============================================================================
with tab_pipe:
    st.subheader("1. 3D Spacing & Contrast Windowing 전처리 결과")
    st.write("Spacing 정렬(`1.25x1.25x5.0mm`) 후 washed-out 명암비를 극복하기 위해 상하위 2% 클리핑 대비 개선을 시각화합니다.")
    
    prep_option = st.radio(
        "시각화 모드 선택:",
        ["기본 전처리 (전체 범위 시각화 - Washed-out 대비)", "대비 개선 전처리 (의료용 Contrast Windowing 적용 - 선명함)"],
        horizontal=True
    )
    img_prep_path = os.path.join(assets_dir, "preprocessing_comparison.png")
    img_prep_contrast_path = os.path.join(assets_dir, "preprocessing_comparison_contrast.png")
    
    if prep_option == "기본 전처리 (전체 범위 시각화 - Washed-out 대비)":
        if os.path.exists(img_prep_path):
            st.image(Image.open(img_prep_path), caption="기본 전처리 결과", use_container_width=True)
        else:
            st.warning("전처리 비교 이미지가 존재하지 않습니다.")
    else:
        if os.path.exists(img_prep_contrast_path):
            st.image(Image.open(img_prep_contrast_path), caption="대비 개선 적용 결과", use_container_width=True)
        else:
            st.warning("대비 개선 이미지가 존재하지 않습니다.")
            
    st.write("---")
    st.subheader("2. 실시간 3D 데이터 증강 및 크롭 패치 대조")
    st.write("배경 노이즈를 피하고 심장 영역을 조밀하게 잘라내는 표적 크롭(`RandCropByPosNegLabeld`) 및 무작위 회전/스케일 증강 대조입니다.")
    
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
        st.image(Image.open(img_path), caption=f"실시간 3D 데이터 증강 및 패스 대조 - {case_option}", use_container_width=True)
    else:
        st.warning(f"선택한 이미지({img_name})가 존재하지 않습니다.")

# ==============================================================================
# TAB 3: Monitor & Model Comparison
# ==============================================================================
with tab_monitor:
    st.subheader("3D U-Net 실시간 학습 곡선 및 성능 교차 검증")
    
    sub_tab_lr, sub_tab_hr, sub_tab_comp = st.tabs([
        "📉 기본 모델 (Low-Res)", 
        "🚀 업스케일 모델 (High-Res)", 
        "📊 두 모델 성능 비교 (Standard vs CCA)"
    ])
    
    # Low-Res Monitor
    with sub_tab_lr:
        history_file_lr = os.path.join(assets_dir, "training_history.json")
        plot_file_lr = os.path.join(assets_dir, "training_curves.png")
        if os.path.exists(history_file_lr) and os.path.exists(plot_file_lr):
            try:
                with open(history_file_lr, "r") as f:
                    history_lr = json.load(f)
                epochs_lr = history_lr.get("epoch", [])
                if epochs_lr:
                    st.metric("최우수 평균 Dice Score", f"{max([d for d in history_lr['val_mean_dice'] if d > 0]):.4f}")
                    st.image(Image.open(plot_file_lr), use_container_width=True)
            except Exception:
                st.warning("Low-Res 기록 파일을 읽어오지 못했습니다.")
        else:
            st.info("Low-Res 훈련 히스토리 파일이 존재하지 않습니다.")
            
    # High-Res Monitor
    with sub_tab_hr:
        history_file_hr = os.path.join(assets_dir, "training_history_hr.json")
        plot_file_hr = os.path.join(assets_dir, "training_curves_hr.png")
        if os.path.exists(history_file_hr) and os.path.exists(plot_file_hr):
            try:
                with open(history_file_hr, "r") as f:
                    history_hr = json.load(f)
                epochs_hr = history_hr.get("epoch", [])
                if epochs_hr:
                    st.metric("최우수 평균 Dice Score", f"{max([d for d in history_hr['val_mean_dice'] if d > 0]):.4f}")
                    st.image(Image.open(plot_file_hr), use_container_width=True)
            except Exception:
                st.warning("High-Res 기록 파일을 읽어오지 못했습니다.")
        else:
            st.info("High-Res 훈련 히스토리 파일이 존재하지 않습니다.")
            
    # Comparison Table (Standard vs CCA)
    with sub_tab_comp:
        st.write("### 📊 다이렉트 교차 검증 비교표 (임상 공간 [1.25, 1.25, 5.0] mm)")
        st.markdown("""
        | 평가지표 (Dice) | 기본 모델 (Low-Res) | 업스케일 모델 (High-Res Standard) | 업스케일 모델 + CCA 필터 (HR + CCA) | 최종 성능 변화량 (Delta) |
        | :--- | :---: | :---: | :---: | :---: |
        | **[테스트] 평균 Dice** | 82.02% | 82.16% | **84.01%** | **+1.99%** 🟢 |
        | [테스트] 우심실 (RV) | 83.12% | 84.99% | **86.44%** | **+3.32%** 🟢 |
        | [테스트] 심근 (MYO) | 74.48% | 74.34% | **75.46%** | **+0.98%** 🟢 |
        | [테스트] 좌심실 (LV) | 88.46% | 87.15% | **90.11%** | **+1.65%** 🟢 (90% 돌파!) |
        |---|---|---|---|---|
        | **[검증] 평균 Dice** | 84.02% | 83.90% | **84.38%** | **+0.36%** 🟢 |
        | [검증] 우심실 (RV) | 85.28% | 87.05% | **87.11%** | **+1.83%** 🟢 |
        | [검증] 심근 (MYO) | 77.05% | 76.14% | **76.49%** | **-0.56%** |
        | [검증] 좌심실 (LV) | 89.74% | 88.50% | **89.54%** | **-0.20%** |
        """)
        st.info("💡 **결과 해석**: 고해상도(HR) 보간을 통해 우심실과 심근의 물리 해상도를 교정한 뒤, CCA(연결 성분 분석) 후처리를 적용함으로써 임상 테스트셋 평균 Dice가 **84% 선을 돌파**했으며, 좌심실 성능이 **90.11%**로 대폭 상승했습니다.")

# ==============================================================================
# TAB 4: Real-time AI Heart Failure Diagnoser
# ==============================================================================
with tab_clinical:
    st.subheader("🏥 AI 기반 3D 심실 구조 세그멘테이션 및 심부전 자동 진단 패널")
    st.write("로컬에 준비된 환자의 3D MRI 데이터를 불러와 실시간으로 3D 분할을 연산하고, 좌심실 박출률(LVEF)을 도출하여 이상 징후를 판독합니다.")
    
    # Load ACDC Splits
    train_files, val_files, test_files = get_acdc_splits(base_dir)
    
    # Build list of selectable patients
    # Extract patient IDs from filename paths
    def extract_patient_info(file_list, group_name):
        patients = {}
        for f in file_list:
            img_path = f["image"]
            filename = os.path.basename(img_path)
            match = re.match(r"(patient\d+)", filename)
            if match:
                pid = match.group(1)
                if pid not in patients:
                    patients[pid] = []
                patients[pid].append(f)
        return {k: v for k, v in patients.items() if len(v) >= 2} # Ensure we have both frames (ED/ES)
        
    val_patients = extract_patient_info(val_files, "Validation")
    test_patients = extract_patient_info(test_files, "Test")
    
    all_selectable_patients = {}
    for pid, files in val_patients.items():
        all_selectable_patients[f"{pid} (Validation)"] = (pid, files)
    for pid, files in test_patients.items():
        all_selectable_patients[f"{pid} (Test)"] = (pid, files)
        
    patient_sel = st.selectbox("🔎 AI 분석을 진행할 환자 케이스 선택:", list(all_selectable_patients.keys()))
    
    col_opt1, col_opt2 = st.columns(2)
    with col_opt1:
        model_type = st.radio("사용할 분할 AI 모델:", ["High-Resolution (고해상도 3D)", "Baseline (저해상도 3D)"], index=0)
    with col_opt2:
        use_cca = st.checkbox("CCA (연결 성분 분석) 후처리 노이즈 필터 적용", value=True)
        
    trigger_btn = st.button("🏥 AI 실시간 판독 및 진단서 발급 시작", type="primary")
    
    # Process Inference
    if trigger_btn or "last_patient" not in st.session_state or st.session_state.last_patient != patient_sel or st.session_state.last_model != model_type or st.session_state.last_cca != use_cca:
        if trigger_btn or "last_patient" not in st.session_state:
            with st.spinner("AI가 환자의 3D MRI 볼륨 데이터를 불러와 수축기/이완기 3차원 분할을 추론하고 있습니다..."):
                pid_code, files = all_selectable_patients[patient_sel]
                
                # Setup models
                if model_type == "High-Resolution (고해상도 3D)":
                    model = load_hr_model(device)
                    is_hr = True
                    roi_size = (128, 128, 16)
                else:
                    model = load_baseline_model(device)
                    is_hr = False
                    roi_size = (128, 128, 8)
                
                patient_results = {}
                
                # Sort files to process both frames
                sorted_files = sorted(files, key=lambda x: x["image"])
                
                for idx, f in enumerate(sorted_files):
                    img_path = f["image"]
                    lbl_path = f["label"]
                    
                    # Read spacing from NIfTI
                    nib_img = nib.load(img_path)
                    spacing = nib_img.header.get_zooms()
                    voxel_vol_mm3 = spacing[0] * spacing[1] * spacing[2]
                    
                    val_transforms = get_val_test_transforms()
                    data_dict = val_transforms({"image": img_path, "label": lbl_path})
                    
                    inputs = data_dict["image"].unsqueeze(0).to(device)
                    labels = data_dict["label"].unsqueeze(0).to(device)
                    
                    if use_cca:
                        from monai.transforms import KeepLargestConnectedComponent, Compose
                        post_pred_lr = Compose([
                            AsDiscrete(to_onehot=4),
                            KeepLargestConnectedComponent(applied_labels=[1, 2, 3], is_onehot=True)
                        ])
                        post_pred_std = Compose([
                            AsDiscrete(argmax=True, to_onehot=4),
                            KeepLargestConnectedComponent(applied_labels=[1, 2, 3], is_onehot=True)
                        ])
                    else:
                        post_pred_lr = AsDiscrete(to_onehot=4)
                        post_pred_std = AsDiscrete(argmax=True, to_onehot=4)
                        
                    with torch.no_grad():
                        if is_hr:
                            from monai.transforms import Spacing
                            resampler_to_hr = Spacing(pixdim=[1.0, 1.0, 2.5], mode="bilinear")
                            resampler_to_lr = Spacing(pixdim=[1.25, 1.25, 5.0], mode="nearest")
                            
                            inputs_hr = resampler_to_hr(data_dict["image"]).unsqueeze(0).to(device)
                            outputs_hr = sliding_window_inference(
                                inputs_hr, 
                                roi_size, 
                                sw_batch_size=4, 
                                predictor=model,
                                overlap=0.5
                            )
                            outputs_argmax_cpu = torch.argmax(outputs_hr, dim=1, keepdim=True).cpu()
                            outputs_lr_argmax = resampler_to_lr(outputs_argmax_cpu[0], output_spatial_shape=labels.shape[2:]).unsqueeze(0).to(device)
                            outputs_processed = post_pred_lr(outputs_lr_argmax[0])
                        else:
                            outputs = sliding_window_inference(
                                inputs, 
                                roi_size, 
                                sw_batch_size=4, 
                                predictor=model,
                                overlap=0.5
                            )
                            outputs_processed = post_pred_std(outputs[0])
                            
                    pred_mask = torch.argmax(outputs_processed, dim=0).cpu().numpy().astype(np.uint8)
                    gt_mask = data_dict["label"][0].cpu().numpy().astype(np.uint8)
                    raw_img = data_dict["image"][0].cpu().numpy()
                    
                    # Calculate volume (mL)
                    rv_vol = (np.sum(pred_mask == 1) * voxel_vol_mm3) / 1000.0
                    myo_vol = (np.sum(pred_mask == 2) * voxel_vol_mm3) / 1000.0
                    lv_vol = (np.sum(pred_mask == 3) * voxel_vol_mm3) / 1000.0
                    
                    patient_results[f"frame_{idx}"] = {
                        "filename": os.path.basename(img_path),
                        "raw_img": raw_img,
                        "gt_mask": gt_mask,
                        "pred_mask": pred_mask,
                        "rv_vol": rv_vol,
                        "myo_vol": myo_vol,
                        "lv_vol": lv_vol,
                        "spacing": spacing
                    }
                    
                # Determine ED vs ES dynamically based on Left Ventricle Volume
                f0_lv = patient_results["frame_0"]["lv_vol"]
                f1_lv = patient_results["frame_1"]["lv_vol"]
                if f0_lv >= f1_lv:
                    ed_key, es_key = "frame_0", "frame_1"
                else:
                    ed_key, es_key = "frame_1", "frame_0"
                    
                # Save to session state
                st.session_state.inf_results = {
                    "ed": patient_results[ed_key],
                    "es": patient_results[es_key]
                }
                st.session_state.last_patient = patient_sel
                st.session_state.last_model = model_type
                st.session_state.last_cca = use_cca
                
    # Display Results if Cached
    if "inf_results" in st.session_state:
        inf = st.session_state.inf_results
        
        # Clinical parameters math
        lvedv = inf["ed"]["lv_vol"]
        lvesv = inf["es"]["lv_vol"]
        lvef = ((lvedv - lvesv) / lvedv) * 100 if lvedv > 0 else 0
        
        rvedv = inf["ed"]["rv_vol"]
        rvesv = inf["es"]["rv_vol"]
        rvef = ((rvedv - rvesv) / rvedv) * 100 if rvedv > 0 else 0
        
        myo_mass = inf["ed"]["myo_vol"] * 1.05
        
        # Metrics Display
        st.write("---")
        st.markdown("### 📊 AI 판독 임상 파라미터 정량화")
        
        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        col_m1.metric("좌심실 이완기말 부피 (LVEDV)", f"{lvedv:.2f} mL")
        col_m2.metric("좌심실 수축기말 부피 (LVESV)", f"{lvesv:.2f} mL")
        
        # LVEF display with warning highlights
        if lvef >= 50:
            col_m3.metric("좌심실 박출률 (LVEF)", f"{lvef:.1f} %", "🟢 정상 수치")
        elif 40 <= lvef < 50:
            col_m3.metric("좌심실 박출률 (LVEF)", f"{lvef:.1f} %", "🟡 경계선 수치", delta_color="off")
        else:
            col_m3.metric("좌심실 박출률 (LVEF)", f"{lvef:.1f} %", "🚨 심부전 의심", delta_color="inverse")
            
        col_m4.metric("추정 심근 질량 (Myo Mass)", f"{myo_mass:.1f} g")
        
        # Clinical Diagnosis Alert Card
        st.markdown("### 🏥 심박출량 분석 및 진단 소견서")
        
        if lvef >= 50:
            st.success(f"""
            **진단 결과: 좌심실 펌프 기능 정상 (LVEF = {lvef:.1f}%)**
            - **소견**: 환자의 좌심실 내경 수축 및 수축력 지표가 정상적인 범위 내에 보존되어 있습니다. 박출률 감소형 심부전(HFrEF) 징후는 나타나지 않습니다.
            """)
        elif 40 <= lvef < 50:
            st.warning(f"""
            **진단 결과: 경계선 박출률 보존 심부전 의심 (LVEF = {lvef:.1f}%)**
            - **소견**: 좌심실 수축력이 정상 하한치에 근접해 있거나 미세하게 저하되었습니다. 경증 수축 기능 부전의 가능성이 있으므로 판독의의 임상 증상 확인 및 3개월 내 추적 심장 초음파/MRI 검사를 권장합니다.
            """)
        else:
            st.error(f"""
            **진단 결과: 🚨 박출률 감소 심부전 의심 (LVEF = {lvef:.1f}%)**
            - **소견**: 좌심실의 쥐어짜는 능력이 40% 미만으로 크게 붕괴된 **박출률 감소 심부전(HFrEF, Heart Failure with Reduced Ejection Fraction)** 소견이 강하게 관찰됩니다. 즉각적인 심장내과 전문의 판독 확인과 함께 혈류 역학적 처방 요법이 필요합니다.
            """)
            
        # Slice Visualizer
        st.write("---")
        st.markdown("### 🖼️ 3D 단면별 세그멘테이션 영상 비교 검증")
        
        col_v1, col_v2 = st.columns([1, 3])
        with col_v1:
            visual_frame = st.radio("시각화할 심장 시점:", ["이완기말 (ED - 피가 가득 참)", "수축기말 (ES - 쥐어짜냄)"])
            active_frame_key = "ed" if visual_frame == "이완기말 (ED - 피가 가득 참)" else "es"
            
            raw_3d = inf[active_frame_key]["raw_img"]
            gt_3d = inf[active_frame_key]["gt_mask"]
            pred_3d = inf[active_frame_key]["pred_mask"]
            num_slices = raw_3d.shape[2]
            
            slice_idx = st.slider("단면 깊이 선택 (Z-Slice):", 0, num_slices - 1, num_slices // 2)
            
        with col_v2:
            fig, ax = plt.subplots(1, 2, figsize=(10, 5))
            
            # Show Ground Truth Overlay
            ax[0].imshow(raw_3d[:, :, slice_idx], cmap="gray")
            overlay_gt = np.zeros((*gt_3d[:, :, slice_idx].shape, 4))
            overlay_gt[gt_3d[:, :, slice_idx] == 1] = [0.0, 0.46, 0.7, 0.4]   # RV (Blue)
            overlay_gt[gt_3d[:, :, slice_idx] == 2] = [1.0, 0.5, 0.0, 0.4]   # MYO (Orange)
            overlay_gt[gt_3d[:, :, slice_idx] == 3] = [0.84, 0.15, 0.16, 0.4] # LV (Red)
            ax[0].imshow(overlay_gt)
            ax[0].set_title("의사의 정답 라벨 (GT Overlay)")
            ax[0].axis("off")
            
            # Show AI Prediction Overlay
            ax[1].imshow(raw_3d[:, :, slice_idx], cmap="gray")
            overlay_pred = np.zeros((*pred_3d[:, :, slice_idx].shape, 4))
            overlay_pred[pred_3d[:, :, slice_idx] == 1] = [0.0, 0.46, 0.7, 0.4]
            overlay_pred[pred_3d[:, :, slice_idx] == 2] = [1.0, 0.5, 0.0, 0.4]
            overlay_pred[pred_3d[:, :, slice_idx] == 3] = [0.84, 0.15, 0.16, 0.4]
            ax[1].imshow(overlay_pred)
            ax[1].set_title("AI 모델 예측 오버레이 (Prediction Overlay)")
            ax[1].axis("off")
            
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
            
        st.info("💡 **색상 매핑 가이드:** 🟦 파란색 = 우심실(RV) | 🟧 주황색 = 심근(MYO) | 🟥 빨간색 = 좌심실(LV)")
