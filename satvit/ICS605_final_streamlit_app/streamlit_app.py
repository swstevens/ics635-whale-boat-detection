import os

import streamlit as st
import torch
from backend.models import FloodSegmentationHead, build_satvit_encoder
from backend.SatViT_inference import (
    predict_new_flood,
    process_satellite_inputs,
    save_flood_prediction_image,
)

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title='SatViT Flood Mapper', page_icon='📡', layout='wide')

# --- UPDATED CUSTOM STYLING ---
st.markdown(
    """
    <style>
    header {visibility: hidden;}
    [data-testid="stHeader"] {background-color: rgba(0,0,0,0) !important;}
    [data-testid="stDecoration"] {display: none;}
    
    .stApp {
        background: linear-gradient(rgba(15, 23, 42, 0.85), rgba(15, 23, 42, 0.85)), 
                    url('https://images.unsplash.com/photo-1451187580459-43490279c0fa?auto=format&fit=crop&w=1920&q=80');
        background-size: cover;
        background-attachment: fixed;
        color: #ffffff !important;
    }

    h1, h2, h3 {
        color: #ffffff !important;
        text-shadow: 2px 2px 4px rgba(0,0,0,1);
    }

    /* Targeting the Sidebar */
    section[data-testid="stSidebar"] {
        background-color: rgba(15, 23, 42, 0.95) !important;
        backdrop-filter: blur(15px);
        border-right: 1px solid rgba(0, 188, 212, 0.2);
    }

    /* Targeting BOTH standard and download buttons */
    .stButton>button, .stDownloadButton>button { 
        background-color: rgba(0, 188, 212, 0.2) !important; /* Darker teal base */
        color: #00bcd4 !important; 
        font-weight: bold !important; 
        border: 1px solid #00bcd4 !important;
        border-radius: 10px !important;
        padding: 0.5rem 1rem !important;
        transition: 0.3s all ease;
        width: 100%;
    }

    /* Hover effect to make it pop */
    .stButton>button:hover, .stDownloadButton>button:hover {
        background-color: #00bcd4 !important;
        color: #0f172a !important;
        box-shadow: 0 0 15px rgba(0, 188, 212, 0.6);
    }

    /* Fixing the white background issue seen in your screenshot */
    .stDownloadButton {
        background-color: transparent !important;
    }

    .main .block-container {
        background-color: rgba(15, 23, 42, 0.5);
        border-radius: 20px;
        padding: 2rem !important;
        border: 1px solid rgba(255, 255, 255, 0.05);
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# --- MODEL LOADING ---
@st.cache_resource
def load_models():
    # Use 'cuda' for NVIDIA, 'mps' for Mac M-series, or 'cpu'
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')

    encoder = build_satvit_encoder(device=device)
    decoder = FloodSegmentationHead(encoder_dim=768, num_classes=1).to(device)

    encoder.load_state_dict(torch.load('final_project_encoder.pth', map_location=device))
    decoder.load_state_dict(torch.load('final_project_decoder.pth', map_location=device))

    encoder.eval()
    decoder.eval()
    return encoder, decoder, device


satvit_model, segmentation_decoder, device = load_models()

# --- SIDEBAR ---
with st.sidebar:
    st.title('🛰️ SatViT Pipeline')
    st.markdown('---')
    overlay_alpha = st.slider('Overlay Transparency', 0.0, 1.0, 0.5)
    st.button('📡 Upload Imagery')
    st.markdown('---')
    # Custom Info Box
    st.markdown(
        f"""
        <div style="background-color: rgba(0, 188, 212, 0.1); padding: 10px; border-radius: 10px; border: 1px solid #00bcd4;">
            <p style="margin:0; color:#00bcd4; font-size: 0.8rem;">Hardware Acceleration</p>
            <p style="margin:0; font-weight: bold;">Running on: {device.type.upper()}</p>
        </div>
    """,
        unsafe_allow_html=True,
    )

# --- MAIN INTERFACE ---
st.title('Geospatial Flood Analysis')
st.write('Detecting flood signatures using fused Sentinel-1 SAR and Sentinel-2 Optical data.')

col1, col2 = st.columns(2)
with col1:
    s1_file = st.file_uploader('Upload Sentinel-1 (Radar)', type=['tif', 'tiff'])
with col2:
    s2_file = st.file_uploader('Upload Sentinel-2 (Optical)', type=['tif', 'tiff'])

if s1_file and s2_file:
    if st.button('Execute Inference Pipeline'):
        os.makedirs('temp_uploads', exist_ok=True)
        os.makedirs('static', exist_ok=True)

        t1 = os.path.join('temp_uploads', s1_file.name)
        t2 = os.path.join('temp_uploads', s2_file.name)

        with open(t1, 'wb') as f:
            f.write(s1_file.getbuffer())
        with open(t2, 'wb') as f:
            f.write(s2_file.getbuffer())

        try:
            with st.status('Analyzing Imagery...', expanded=True) as status:
                prep = process_satellite_inputs(t1, t2, target_size=256)
                f_map = predict_new_flood(prep, satvit_model, segmentation_decoder, device)

                st.session_state['current_flood_map'] = f_map
                st.session_state['last_s2_path'] = t2
                status.update(label='Analysis Complete!', state='complete')

            if os.path.exists(t1):
                os.remove(t1)
        except Exception as e:
            st.error(f'Error: {e}')

# --- RENDER RESULTS ---
if 'current_flood_map' in st.session_state:
    out_path = os.path.join('static', 'active_render.png')
    save_flood_prediction_image(
        st.session_state['last_s2_path'],
        st.session_state['current_flood_map'],
        out_path,
        alpha=overlay_alpha,
    )

    st.markdown('---')
    st.subheader('Model Visualization')
    st.image(out_path, use_container_width=True)

    # The Download Button is now styled to be dark and match the theme
    with open(out_path, 'rb') as file:
        st.download_button(
            label='💾 Save Analysis Result',
            data=file,
            file_name='oahu_flood_analysis.png',
            mime='image/png',
        )
