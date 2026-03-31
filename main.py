import streamlit as st
import requests
from PIL import Image
import plotly.graph_objects as go

# ===================== CONFIG =====================
API_URL = "https://junaid17-damagelensai.hf.space"

st.set_page_config(
    page_title="Car Damage AI",
    layout="wide",
    page_icon="🚗",
    initial_sidebar_state="collapsed"
)

# ===================== ADVANCED UI & ANIMATIONS =====================
st.markdown("""
<style>
    /* Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;800&display=swap');

    /* Base Theme */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
        background-color: #09090b !important; /* Deeper slate for higher contrast */
        color: #e2e8f0 !important;
    }
    
    /* Clean up default Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {background-color: transparent !important;}

    /* === Animations === */
    @keyframes slideUpFade {
        0% { opacity: 0; transform: translateY(30px); }
        100% { opacity: 1; transform: translateY(0); }
    }
    
    @keyframes pulseGlow {
        0% { box-shadow: 0 0 10px rgba(0, 198, 255, 0.2); }
        50% { box-shadow: 0 0 25px rgba(0, 198, 255, 0.6); }
        100% { box-shadow: 0 0 10px rgba(0, 198, 255, 0.2); }
    }

    @keyframes shimmer {
        0% { background-position: -200% center; }
        100% { background-position: 200% center; }
    }

    /* === Custom Components === */
    .card {
        background: #18181b; 
        border: 1px solid #27272a;
        padding: 24px;
        border-radius: 16px;
        box-shadow: 0px 8px 24px rgba(0,0,0,0.4);
        animation: slideUpFade 0.6s ease-out forwards;
        transition: all 0.3s ease;
        margin-bottom: 1rem;
        width: 100%;
        box-sizing: border-box;
    }
    
    .card:hover {
        transform: translateY(-4px);
        box-shadow: 0px 12px 32px rgba(0, 198, 255, 0.15);
        border-color: #3f3f46;
    }

    /* Primary Button Styling */
    .stButton>button {
        background: linear-gradient(135deg, #00c6ff 0%, #0072ff 100%);
        color: white !important;
        border: none;
        border-radius: 12px;
        height: 3.5em;
        font-size: 16px;
        font-weight: 600;
        width: 100%; 
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(0, 114, 255, 0.3);
    }
    .stButton>button:hover {
        transform: scale(1.02);
        animation: pulseGlow 1.5s infinite;
    }

    /* Typography */
    h1, h2, h3 {
        font-weight: 800 !important;
        letter-spacing: -0.5px;
    }
    
    /* Shimmering Main Title */
    .shimmer-text {
        text-align: center;
        font-size: 3rem;
        font-weight: 800;
        background: linear-gradient(90deg, #e2e8f0 0%, #ffffff 25%, #00c6ff 50%, #e2e8f0 75%, #e2e8f0 100%);
        background-size: 200% auto;
        color: transparent;
        -webkit-background-clip: text;
        background-clip: text;
        animation: shimmer 4s linear infinite;
        margin-bottom: 0.5rem;
    }

    .big-text {
        font-size: 2.5rem;
        font-weight: 800;
        background: -webkit-linear-gradient(45deg, #00c6ff, #0072ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }

    /* Custom Progress Bar */
    .progress-wrapper {
        background: #27272a;
        border-radius: 20px;
        overflow: hidden;
        height: 12px;
        margin-top: 8px;
        margin-bottom: 16px;
        box-shadow: inset 0 2px 4px rgba(0,0,0,0.5);
    }
    .progress-fill {
        height: 100%;
        background: linear-gradient(90deg, #00c6ff, #0072ff);
        border-radius: 20px;
        transition: width 1.5s cubic-bezier(0.22, 1, 0.36, 1);
    }

    /* Image styling & Hover */
    img {
        border-radius: 12px;
        object-fit: cover;
        transition: transform 0.3s ease;
    }
    .card img:hover {
        transform: scale(1.01);
    }

    /* ===================== MOBILE RESPONSIVENESS ===================== */
    @media (max-width: 768px) {
        .card {
            padding: 16px;
        }
        .shimmer-text {
            font-size: 2.2rem;
        }
        .big-text {
            font-size: 2rem;
        }
        .stButton>button {
            height: 3em;
            font-size: 15px;
        }
        /* Make detection logs tighter on mobile */
        .detection-log {
            padding: 8px !important;
            font-size: 0.9em;
        }
    }
</style>
""", unsafe_allow_html=True)

# ===================== HELPERS =====================
def plot_bar_chart(data, title=""):
    labels = list(data.keys())
    values = list(data.values())

    fig = go.Figure(data=[go.Bar(
        x=labels, 
        y=values, 
        marker_color='#00c6ff',
        marker_line_color='#0072ff',
        marker_line_width=1.5,
        opacity=0.85
    )])
    
    fig.update_layout(
        template="plotly_dark",
        title=title,
        xaxis_title="Classes",
        yaxis_title="Probability",
        paper_bgcolor='rgba(0,0,0,0)', 
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=10, r=10, t=40, b=20),
        font=dict(family="Inter", size=12, color="#a1a1aa")
    )
    # Hide the modebar for a cleaner mobile look
    return fig

def call_api(endpoint, file_bytes):
    files = {"image": file_bytes}
    res = requests.post(f"{API_URL}{endpoint}", files=files)
    if res.status_code != 200:
        raise Exception(res.text)
    return res.json()

def call_gradcam(file_bytes):
    files = {"file": file_bytes}
    res = requests.post(f"{API_URL}/predict", files=files)
    if res.status_code != 200:
        raise Exception(res.text)
    return res.json()

def call_yolo(file_bytes):
    files = {"file": file_bytes}
    res = requests.post(f"{API_URL}/predict/yolo", files=files)
    if res.status_code != 200:
        raise Exception(res.text)
    return res.json()

# ===================== HEADER =====================
st.markdown('<div class="shimmer-text">🚗 Car Damage AI</div>', unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #a1a1aa; font-size: 1.1rem; margin-bottom: 2.5rem;'>Fusion Intelligence: ResNet + DeiT + YOLO</p>", unsafe_allow_html=True)

# ===================== MAIN APP =====================
main_container = st.container()

with main_container:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload Vehicle Image", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
    st.markdown('</div>', unsafe_allow_html=True)

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        file_bytes = uploaded_file.getvalue()

        # Mobile-friendly columns
        col1, col2 = st.columns([1.2, 1], gap="large")

        with col1:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.image(image, caption="Uploaded Image", use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with col2:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown("### ⚙️ Analysis Settings")
            mode = st.selectbox(
                "Select Prediction Engine",
                ["Fusion (Recommended)", "ResNet Only", "DeiT Only"],
                index=0
            )
            
            mode_map = {"Fusion (Recommended)": "Fusion", "ResNet Only": "ResNet", "DeiT Only": "DeiT"}
            selected_mode = mode_map[mode]
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            # --- COLD START WARNING ---
            st.info("⏱️ **Note:** The first analysis may take up to **30-60 seconds** while the AI models warm up. Subsequent requests will be much faster!")
            
            predict_btn = st.button("🚀 Run AI Analysis", use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        # ===================== EXECUTION & RESULTS =====================
        if predict_btn:
            st.divider() 
            
            try:
                with st.status("🧠 Analyzing Vehicle Damage...", expanded=True) as status:
                    st.write("Extracting features...")
                    if selected_mode == "Fusion":
                        pred_data = call_api("/predict/fusion", file_bytes)
                    elif selected_mode == "ResNet":
                        pred_data = call_api("/predict/resnet", file_bytes)
                    else:
                        pred_data = call_api("/predict/deit", file_bytes)
                    
                    st.write("Generating attention maps...")
                    cam_data = call_gradcam(file_bytes)
                    
                    st.write("Running YOLO object detection...")
                    yolo_data = call_yolo(file_bytes)
                    
                    status.update(label="✅ Analysis Complete!", state="complete", expanded=False)

                # --- RESULTS TABS ---
                # Streamlit automatically handles tab responsiveness on mobile by enabling horizontal scrolling
                tab1, tab2, tab3 = st.tabs(["📊 Prediction Output", "👀 Model Attention", "🎯 Damage Localization"])

                # TAB 1: PREDICTION
                with tab1:
                    st.markdown('<div class="card">', unsafe_allow_html=True)
                    if selected_mode == "Fusion":
                        st.markdown(f'<div class="big-text">{pred_data["final_prediction"]}</div>', unsafe_allow_html=True)
                        
                        confidence = pred_data["final_confidence"] * 100
                        st.markdown(f"**Confidence Score: {confidence:.2f}%**")
                        st.markdown(
                            f"""
                            <div class="progress-wrapper">
                                <div class="progress-fill" style="width:{confidence}%"></div>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )

                        st.markdown("### Fused Probabilities")
                        st.plotly_chart(plot_bar_chart(pred_data["fused_output"]), use_container_width=True, config={'displayModeBar': False})

                        with st.expander("🔍 View Individual Network Details"):
                            st.plotly_chart(plot_bar_chart(pred_data["resnet_output"], "ResNet Output"), use_container_width=True, config={'displayModeBar': False})
                            st.plotly_chart(plot_bar_chart(pred_data["deit_output"], "DeiT Output"), use_container_width=True, config={'displayModeBar': False})

                    else:
                        st.markdown(f"### {selected_mode} Output")
                        st.plotly_chart(plot_bar_chart(pred_data), use_container_width=True, config={'displayModeBar': False})
                    st.markdown('</div>', unsafe_allow_html=True)

                # TAB 2: EXPLAINABILITY
                with tab2:
                    st.markdown('<div class="card">', unsafe_allow_html=True)
                    st.markdown("### How the AI sees the damage")
                    
                    # Columns stack automatically on mobile
                    c1, c2, c3 = st.columns(3)
                    with c1:
                        st.image(f"{API_URL}{cam_data['original_image']}", caption="Original", use_container_width=True)
                    with c2:
                        st.image(f"{API_URL}{cam_data['resnet_viz']}", caption="ResNet Focus", use_container_width=True)
                    with c3:
                        st.image(f"{API_URL}{cam_data['deit_viz']}", caption="DeiT Focus", use_container_width=True)
                    st.markdown('</div>', unsafe_allow_html=True)

                # TAB 3: YOLO LOCALIZATION
                with tab3:
                    y_col1, y_col2 = st.columns([1.5, 1], gap="medium")
                    
                    with y_col1:
                        st.markdown('<div class="card">', unsafe_allow_html=True)
                        st.image(f"{API_URL}{yolo_data['yolo_image']}", caption="YOLO Bounding Boxes", use_container_width=True)
                        st.markdown('</div>', unsafe_allow_html=True)

                    with y_col2:
                        st.markdown('<div class="card">', unsafe_allow_html=True)
                        st.markdown("### Detection Log")
                        
                        if yolo_data["total_detections"] == 0:
                            st.info("🟢 No specific damage bounding boxes detected.")
                        else:
                            st.warning(f"🔴 Found **{yolo_data['total_detections']}** damage region(s).")
                            for i, det in enumerate(yolo_data["detections"]):
                                st.markdown(
                                    f"""
                                    <div class="detection-log" style="background: #27272a; padding: 12px; border-radius: 8px; margin-bottom: 10px; border-left: 4px solid #00c6ff; box-shadow: 0 2px 4px rgba(0,0,0,0.2);">
                                        <b style="color: #e2e8f0;">Region {i+1}</b><br>
                                        <span style="color: #a1a1aa; font-size: 0.9em;">Confidence: {(det['confidence']*100):.1f}%</span>
                                    </div>
                                    """, 
                                    unsafe_allow_html=True
                                )
                        st.markdown('</div>', unsafe_allow_html=True)

            except Exception as e:
                st.error(f"⚠️ Error communicating with AI server: {str(e)}")