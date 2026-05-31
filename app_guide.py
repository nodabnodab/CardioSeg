import os
import streamlit as st
from PIL import Image

# Set Page Config
st.set_page_config(
    page_title="CardioSeg3D: Cardiac MRI Guide",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling for Sleek Aesthetic
st.markdown("""
<style>
    .main-title {
        font-size: 2.8rem;
        font-weight: 800;
        color: #ff4b4b;
        margin-bottom: 0.5rem;
    }
    .sub-title {
        font-size: 1.2rem;
        color: #555;
        margin-bottom: 2rem;
    }
    .metric-box {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #ff4b4b;
        margin-bottom: 1rem;
    }
    .calc-box {
        background-color: #e8f4fd;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
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

# Sidebar
st.sidebar.title("🫀 CardioSeg3D")
st.sidebar.markdown("""
**심장 MRI 다중 구조 분할 및 임상 지표 자동 정량화 파이프라인**
***
*이 가이드는 비전공자 및 판독의의 시야에서 영상의 해석 방식과 임상적 의의를 돕기 위해 작성되었습니다.*
""")
st.sidebar.info("💡 **가상환경 정보**\n- Python 3.11.9\n- Nibabel / SimpleITK\n- Streamlit")

# Main Content
st.markdown('<h1 class="main-title">🫀 심장 MRI 분할 & 임상 분석 가이드</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">비전공자도 쉽게 이해할 수 있는 의료 인공지능(AI)과 심장 MRI 분석의 원리</p>', unsafe_allow_html=True)

# Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "💡 심장의 구조 & 핵심 타겟", 
    "🖼️ MRI 데이터 시각화", 
    "📈 임상 지표 (박출률 & 질량)", 
    "🧮 복셀 부피 계산기"
])

# Tab 1: Structure & Targets
with tab1:
    st.subheader("1. 뼈대 이해하기: 심장의 기본 구조와 3대 핵심 타겟")
    st.write(
        "개발할 인공지능(AI)은 심장 MRI 사진을 보고 **심장의 특정 부위 3곳**을 자동으로 구분해서 색칠하는 법을 배웁니다."
    )
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="metric-box">
            <h4>🔵 우심실 (Right Ventricle - RV)</h4>
            <p><b>마스크 번호: 1번</b></p>
            <p>몸을 한 바퀴 돌며 산소를 소모하고 돌아온 피를 받아, <b>폐(Lung)로 뿜어내어 다시 산소를 채워오게 하는 펌프</b>입니다.</p>
        </div>
        """, unsafe_allow_html=True)
        
    with col2:
        st.markdown("""
        <div class="metric-box" style="border-left-color: #ff7f0e;">
            <h4>🟠 심근 (Myocardium - MYO)</h4>
            <p><b>마스크 번호: 2번</b></p>
            <p>심장을 둘러싸고 있는 <b>두꺼운 근육 벽</b>입니다. 이 근육이 강력하게 수축(쥐어짜기)과 이완(부풀기)을 반복하며 피를 이동시킵니다.</p>
        </div>
        """, unsafe_allow_html=True)
        
    with col3:
        st.markdown("""
        <div class="metric-box" style="border-left-color: #d62728;">
            <h4>🔴 좌심실 (Left Ventricle - LV)</h4>
            <p><b>마스크 번호: 3번</b></p>
            <p>폐에서 산소를 가득 채워온 피를 받아 <b>온몸(뇌, 주요 장기 등)으로 뿜어내 주는 가장 힘이 센 핵심 펌프</b>입니다.</p>
        </div>
        """, unsafe_allow_html=True)

# Tab 2: Visualizations
with tab2:
    st.subheader("2. 우리가 다룰 데이터 눈으로 직접 보기")
    st.write("3D NIfTI 파일에서 심장 부위가 가장 잘 보이는 중간 단면(Slice 5)을 시각화한 결과입니다.")
    
    col1, col2, col3 = st.columns(3)
    
    if os.path.exists(img_raw_path) and os.path.exists(img_mask_path) and os.path.exists(img_overlay_path):
        with col1:
            st.image(Image.open(img_raw_path), caption="1. 순수 MRI 영상 (흑백)", use_container_width=True)
            st.markdown("* 흑백 음영으로 나타나는 오리지널 MRI 영상입니다. 가운데 부분에 동그란 모양의 심장 구조가 보입니다.")
            
        with col2:
            st.image(Image.open(img_mask_path), caption="2. 의사가 색칠해 둔 정답 마스크", use_container_width=True)
            st.markdown("* **파란색**: 우심실(RV)\n* **주황색**: 심근(MYO)\n* **빨간색**: 좌심실(LV)")
            
        with col3:
            st.image(Image.open(img_overlay_path), caption="3. MRI 영상 위에 마스크를 얹은 모습", use_container_width=True)
            st.markdown("* 영상 위에 마스크를 투명하게 입힌 최종 비교 모습입니다. AI는 이 매핑 관계를 학습합니다.")
    else:
        st.error("이미지 파일이 `assets/` 폴더에 존재하지 않습니다. 먼저 이미지 생성 스크립트가 실행되었는지 확인해 주세요.")

    st.markdown("---")
    st.subheader("⚙️ 3D 데이터 전처리 (Preprocessing) 결과 비교")
    st.write("모델 학습을 위해 해상도 통일(Spacing), 전경 추출(Crop), 밝기 정규화(Normalize)를 수행한 결과입니다.")
    
    prep_option = st.radio(
        "시각화 옵션 선택:",
        ["기본 전처리 (전체 범위 시각화)", "대비 개선 전처리 (의료용 Contrast Windowing 적용)"],
        horizontal=True
    )
    
    img_prep_path = os.path.join(assets_dir, "preprocessing_comparison.png")
    img_prep_contrast_path = os.path.join(assets_dir, "preprocessing_comparison_contrast.png")
    
    if prep_option == "기본 전처리 (전체 범위 시각화)":
        if os.path.exists(img_prep_path):
            st.image(Image.open(img_prep_path), caption="기본 전처리 결과 (2x2)", use_container_width=True)
            st.info("💡 **특징:** 밝기 정규화(평균 0, 표준편차 1) 후 단순히 최소/최대값 범위로 그려 대비가 분산되어(Washed-out) 약간 흐려 보입니다.")
        else:
            st.warning("전처리 비교 이미지가 존재하지 않습니다. `visualize_preprocessing.py`를 실행해 주세요.")
    else:
        if os.path.exists(img_prep_contrast_path):
            st.image(Image.open(img_prep_contrast_path), caption="대비 개선(Contrast Windowing) 적용 결과 (2x2)", use_container_width=True)
            st.success("✨ **특징:** 밝기 노이즈(상위 2%, 하위 2%)를 깎아내고 중심부 장기 대비만 극대화하는 의료용 Windowing 기법을 적용하여 경계선과 픽셀들이 선명하게 복원되었습니다.")
        else:
            st.warning("대비 개선 이미지가 존재하지 않습니다. `visualize_tuning.py`를 실행해 주세요.")

    st.markdown("---")
    st.subheader("🔄 3D 데이터 증강 (Data Augmentation) 및 패치 크롭 검증")
    st.write("""
    모델 학습 시 VRAM 오버플로우를 예방하고 데이터 다양성을 높이기 위해 실시간으로 적용되는 **3D 데이터 증강 및 패치 크롭** 시각화 결과입니다.
    하나의 환자 볼륨에서 심장 부위를 위주로 **4개의 서로 다른 128x128x8 크기의 3D 패치**를 무작위로 추출하며, 회전/줌/밝기 변형이 실시간으로 동반됩니다.
    """)
    
    img_aug_path = os.path.join(assets_dir, "augmentation_verification.png")
    if os.path.exists(img_aug_path):
        st.image(Image.open(img_aug_path), caption="실시간 3D 데이터 증강 및 패치 추출 결과 (4개 패치 비교)", use_container_width=True)
        st.info("""
        💡 **확인 포인트:**
        - **회전/명암 변동:** 각 패치마다 무작위 회전(Rotated) 및 밝기 스케일링이 들어가 형태와 음영이 미세하게 다릅니다.
        - **표적 중심 추출:** 배경만 잘라내지 않고, 타겟(우심실/좌심실/심근)이 일부라도 포함된 영역 위주로 똑똑하게 잘라내어 학습 효율을 극대화합니다.
        """)
    else:
        st.warning("데이터 증강 검증 이미지가 존재하지 않습니다. `verify_augmentation.py`를 실행해 주세요.")

# Tab 3: Clinical Metrics
with tab3:
    st.subheader("3. 핵심 의학 용어 & 임상 지표 설명")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### ① 수축기와 이완기")
        st.write("""
        * **이완기 끝 (End-Diastole, ED)**: 심장에 피가 가득 차서 **가장 풍선처럼 부풀어 올랐을 때**의 시점입니다. (MRI 상에서 빨간색 면적이 가장 넓음)
        * **수축기 끝 (End-Systole, ES)**: 심장이 피를 밖으로 다 쥐어짜서 **가장 쪼그라들었을 때**의 시점입니다. (MRI 상에서 빨간색 면적이 가장 좁음)
        """)
        st.info("💡 **이 2개 프레임만 쓰는 이유:** 심장의 펌프 성능을 계산하려면 '가장 컸을 때의 부피'와 '가장 작아졌을 때의 부피' 두 가지만 알면 되기 때문입니다.")
        
    with col2:
        st.markdown("### ② 박출률 (Ejection Fraction, EF)")
        st.write("""
        심장이 한 번 뛸 때 **좌심실에서 몸 전체로 내보내는 피의 비율(%)**을 의미하며, 심장 건강 평가의 핵심 지표입니다.
        """)
        st.latex(r"\text{EF (\%)} = \frac{\text{이완기 부피(EDV)} - \text{수축기 부피(ESV)}}{\text{이완기 부피(EDV)}} \times 100")
        
        st.markdown("""
        * **정상 수치**: **55% ~ 70%** (정상)
        * **위험 수치**: **40% 이하** (심장이 피를 짜내는 힘이 부족한 **심부전** 의심)
        """)
        
    st.markdown("---")
    st.markdown("### ③ 심근 질량 (Myocardial Mass)")
    st.write("""
    심장 근육(주황색 띠)의 실제 무게(g 단위)입니다. 
    심장이 좋지 않아 비정상적으로 벽이 두꺼워지는 **심근비대증** 등을 진단할 때 사용합니다.
    """)
    st.latex(r"\text{Myocardial Mass (g)} = \text{심근 부피(Myocardium Volume, ml)} \times 1.05 \text{ g/ml (심근 밀도)}")

# Tab 4: Interactive Calculator
with tab4:
    st.subheader("4. 의료 데이터의 부피 계산 원리 (NIfTI Spacing)")
    st.write("""
    일반 이미지와 달리 의료 표준 포맷인 **NIfTI**는 화소(Pixel) 하나가 실제 환자의 몸 안에서 가로, 세로, 높이 몇 mm 크기인지를 나타내는 **Spacing** 메타데이터를 포함합니다.
    이 3D 화소를 **복셀(Voxel)**이라고 부릅니다.
    """)
    
    st.markdown("### 🧮 실시간 복셀 부피 계산기")
    st.write("아래 슬라이더를 조정하여 복셀 개수와 Spacing에 따라 실제 물리적인 부피(ml)가 어떻게 변하는지 확인해 보세요.")
    
    calc_col1, calc_col2 = st.columns([1, 1])
    
    with calc_col1:
        dx = st.slider("가로 복셀 크기 (dx, mm)", min_value=0.5, max_value=3.0, value=1.5625, step=0.1)
        dy = st.slider("세로 복셀 크기 (dy, mm)", min_value=0.5, max_value=3.0, value=1.5625, step=0.1)
        dz = st.slider("Z축 슬라이스 두께 (dz, mm)", min_value=1.0, max_value=15.0, value=10.0, step=0.5)
        voxel_count = st.number_input("인공지능이 검출한 좌심실(LV) 복셀 개수 (개)", min_value=100, max_value=50000, value=5000, step=100)
        
    with calc_col2:
        # Volume math
        voxel_vol_mm3 = dx * dy * dz
        total_vol_mm3 = voxel_count * voxel_vol_mm3
        total_vol_ml = total_vol_mm3 / 1000.0  # 1 ml = 1000 mm^3
        
        st.markdown(f"""
        <div class="calc-box">
            <h4>📊 계산 결과</h4>
            <hr>
            <p>• <b>복셀 1개의 부피:</b> {voxel_vol_mm3:.4f} mm³</p>
            <p>• <b>총 부피 (mm³):</b> {total_vol_mm3:,.2f} mm³</p>
            <p style="font-size: 1.3rem; color: #1f77b4;">• <b>실제 환자의 심장 부피 (ml):</b> <strong>{total_vol_ml:.2f} ml</strong></p>
        </div>
        """, unsafe_allow_html=True)
        
        st.info("""
        📝 **실제 임상 적용 예시:**
        - 만약 **이완기(ED)** 상태의 복셀이 6,500개라면 ➡️ EDV = **158.70 ml**
        - **수축기(ES)** 상태의 복셀이 2,500개라면 ➡️ ESV = **61.04 ml**
        - 위 계산기로부터 도출된 박출률(EF)은 **61.5%** (정상 범위)가 됩니다.
        """)
