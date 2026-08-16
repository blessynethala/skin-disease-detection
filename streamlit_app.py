import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

IMG_SIZE = 224
CLASSES = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']

DISEASE_INFO = {
    'akiec': {
        'name': 'Actinic Keratoses',
        'subtitle': 'Pre-cancerous',
        'description': 'A rough, scaly patch caused by years of sun exposure. Can potentially develop into skin cancer if untreated.',
        'risk': 'Moderate',
        'risk_color': '#F59E0B',
        'stage': 'Pre-cancerous'
    },
    'bcc': {
        'name': 'Basal Cell Carcinoma',
        'subtitle': '',
        'description': 'The most common type of skin cancer. Grows slowly and rarely spreads, but needs treatment.',
        'risk': 'High',
        'risk_color': '#EF4444',
        'stage': 'Cancerous'
    },
    'bkl': {
        'name': 'Benign Keratosis',
        'subtitle': '',
        'description': 'A non-cancerous skin growth, often appearing as a waxy or scaly raised patch.',
        'risk': 'Low',
        'risk_color': '#10B981',
        'stage': 'Benign'
    },
    'df': {
        'name': 'Dermatofibroma',
        'subtitle': '',
        'description': 'A common benign skin nodule, usually harmless, often caused by minor skin injury.',
        'risk': 'Low',
        'risk_color': '#10B981',
        'stage': 'Benign'
    },
    'mel': {
        'name': 'Melanoma',
        'subtitle': '',
        'description': 'The most serious and dangerous type of skin cancer. Can spread quickly if not treated early.',
        'risk': 'Critical',
        'risk_color': '#B91C1C',
        'stage': 'Cancerous'
    },
    'nv': {
        'name': 'Melanocytic Nevi',
        'subtitle': 'Common Mole',
        'description': 'A common, usually harmless mole made of pigment-producing cells.',
        'risk': 'Low',
        'risk_color': '#10B981',
        'stage': 'Benign'
    },
    'vasc': {
        'name': 'Vascular Lesions',
        'subtitle': '',
        'description': 'Skin marks caused by blood vessel abnormalities, such as birthmarks or angiomas.',
        'risk': 'Low',
        'risk_color': '#10B981',
        'stage': 'Benign'
    }
}


@st.cache_resource
def load_trained_model():
    return tf.keras.models.load_model('final_skin_disease_model.keras')


def predict(image, model):
    img = image.convert('RGB').resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(img)
    img_array = tf.keras.applications.efficientnet.preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)
    preds = model.predict(img_array, verbose=0)[0]
    return preds


st.set_page_config(page_title="SkinSense AI", page_icon="🩺", layout="centered")

# ---- Custom CSS styling ----
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Poppins', sans-serif;
    }

    .stApp {
        background: radial-gradient(circle at 15% 10%, #EEF2FF 0%, transparent 45%),
                    radial-gradient(circle at 85% 15%, #F5F3FF 0%, transparent 45%),
                    radial-gradient(circle at 50% 90%, #EFF6FF 0%, transparent 50%),
                    linear-gradient(160deg, #F8FAFF 0%, #F5F7FF 50%, #FAF5FF 100%);
    }
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #EEF2FF, #F5F3FF);
    }
    .main-header {
        text-align: center;
        padding: 1.5rem 0 0.3rem 0;
    }
    .main-header h1 {
        font-weight: 700;
        font-size: 2.4rem;
        margin-bottom: 0.3rem;
        background: linear-gradient(90deg, #6366F1, #8B5CF6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .subtitle-black {
        text-align: center;
        color: #1F2937;
        font-size: 1rem;
        font-weight: 500;
        margin-bottom: 0.3rem;
    }
    .subtitle {
        text-align: center;
        color: #6366F1;
        font-size: 0.9rem;
        font-weight: 500;
        margin-bottom: 1.5rem;
    }
    .result-card {
        background: #FFFFFF;
        border-radius: 18px;
        padding: 1.6rem;
        margin-top: 1rem;
        border: 1px solid #E5E7EB;
        box-shadow: 0 8px 24px rgba(99, 102, 241, 0.08);
    }
    .risk-badge {
        display: inline-block;
        padding: 0.3rem 1rem;
        border-radius: 20px;
        color: white;
        font-weight: 600;
        font-size: 0.8rem;
        margin-right: 0.5rem;
    }
    .stage-badge {
        display: inline-block;
        padding: 0.3rem 1rem;
        border-radius: 20px;
        background: #EEF2FF;
        color: #6366F1;
        font-weight: 600;
        font-size: 0.8rem;
    }
    .disclaimer-box {
        background: #FFFBEB;
        border-left: 4px solid #F59E0B;
        border-radius: 10px;
        padding: 1rem 1.2rem;
        margin-top: 1.5rem;
        font-size: 0.9rem;
        box-shadow: 0 4px 12px rgba(245, 158, 11, 0.08);
    }
    .footer-box {
        text-align: center;
        color: #9CA3AF;
        font-size: 0.8rem;
        margin-top: 2.5rem;
        padding-top: 1rem;
        border-top: 1px solid #E5E7EB;
    }
    .stProgress > div > div > div {
        border-radius: 10px !important;
        background-color: transparent !important;
        background-image: linear-gradient(90deg, #6366F1, #8B5CF6) !important;
    }
    .stProgress > div > div {
        background-color: #E5E7EB !important;
        border-radius: 10px !important;
    }
    div[data-testid="stFileUploader"] {
        background: #FFFFFF;
        border-radius: 14px;
        padding: 0.5rem;
        box-shadow: 0 4px 14px rgba(99, 102, 241, 0.06);
    }
</style>
""", unsafe_allow_html=True)

# ---- Header ----
st.markdown("""
<div class="main-header">
    <h1>🩺 SkinSense AI</h1>
</div>
<p class="subtitle-black">An Intelligent Skin Disease Detection System Using Deep Learning</p>
<p class="subtitle">EfficientNetB0 · Transfer Learning · HAM10000 Dataset · 7 Lesion Classes</p>
""", unsafe_allow_html=True)

st.markdown(
    'Upload a clear, well-lit image of a skin lesion to get an '
    '<span style="color:#3B82F6; font-weight:600;">AI-assisted preliminary screening</span>. '
    'This is a portfolio/learning project — <strong>not</strong> a substitute for professional medical diagnosis.',
    unsafe_allow_html=True
)

model = load_trained_model()

uploaded_file = st.file_uploader("📤 Upload Skin Lesion Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    col1, col2 = st.columns([1, 1])

    image = Image.open(uploaded_file)
    with col1:
        st.image(image, caption="Uploaded Image", use_container_width=True)

    with st.spinner("🔬 Analyzing image..."):
        preds = predict(image, model)

    top_idx = int(np.argmax(preds))
    top_class = CLASSES[top_idx]
    top_conf = float(preds[top_idx]) * 100
    info = DISEASE_INFO[top_class]

    with col2:
        st.markdown(f"""
        <div class="result-card">
            <h3 style="margin-top:0; color:#1F2937;">{info['name']}</h3>
            <span class="risk-badge" style="background-color:{info['risk_color']};">
                {info['risk']} Risk
            </span>
            <span class="stage-badge">{info['stage']}</span>
            <p style="margin-top:1rem; color:#4B5563;">{info['description']}</p>
        </div>
        """, unsafe_allow_html=True)
        st.metric("Model Confidence", f"{top_conf:.1f}%")

    st.markdown("### 📊 Confidence Across All Classes")
    sorted_indices = np.argsort(preds)[::-1]
    for idx in sorted_indices:
        cls = CLASSES[idx]
        conf = float(preds[idx])
        st.write(f"**{DISEASE_INFO[cls]['name']}** — {conf*100:.1f}%")
        st.progress(conf)

    st.markdown("""
    <div class="disclaimer-box">
        ⚠️ <strong>Disclaimer:</strong> This is an AI-assisted preliminary screening tool, NOT a medical
        diagnosis. Please consult a certified dermatologist for accurate diagnosis and treatment.
        This model achieves ~70% overall test accuracy on held-out data; performance varies by disease type.
    </div>
    """, unsafe_allow_html=True)
else:
    st.info("👆 Upload an image above to get started.")

st.markdown("""
<div class="footer-box">
    SkinSense AI · Built with EfficientNetB0 & Streamlit · Portfolio Project<br>
    Disclaimer: Not a substitute for professional medical advice.
</div>
""", unsafe_allow_html=True)
