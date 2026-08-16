import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

IMG_SIZE = 224
CLASSES = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']

DISEASE_INFO = {
    'akiec': {
        'name': 'Actinic Keratoses (Pre-cancerous)',
        'description': 'A rough, scaly patch caused by years of sun exposure. Can potentially develop into skin cancer if untreated.',
        'risk': 'Moderate',
        'risk_color': '#FFA500'
    },
    'bcc': {
        'name': 'Basal Cell Carcinoma',
        'description': 'The most common type of skin cancer. Grows slowly and rarely spreads, but needs treatment.',
        'risk': 'High',
        'risk_color': '#FF4B4B'
    },
    'bkl': {
        'name': 'Benign Keratosis',
        'description': 'A non-cancerous skin growth, often appearing as a waxy or scaly raised patch.',
        'risk': 'Low',
        'risk_color': '#00C853'
    },
    'df': {
        'name': 'Dermatofibroma',
        'description': 'A common benign skin nodule, usually harmless, often caused by minor skin injury.',
        'risk': 'Low',
        'risk_color': '#00C853'
    },
    'mel': {
        'name': 'Melanoma',
        'description': 'The most serious and dangerous type of skin cancer. Can spread quickly if not treated early.',
        'risk': 'Critical',
        'risk_color': '#D32F2F'
    },
    'nv': {
        'name': 'Melanocytic Nevi (Common Mole)',
        'description': 'A common, usually harmless mole made of pigment-producing cells.',
        'risk': 'Low',
        'risk_color': '#00C853'
    },
    'vasc': {
        'name': 'Vascular Lesions',
        'description': 'Skin marks caused by blood vessel abnormalities, such as birthmarks or angiomas.',
        'risk': 'Low',
        'risk_color': '#00C853'
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


st.set_page_config(page_title="AI Skin Disease Detection", page_icon="🩺", layout="centered")

# ---- Custom CSS styling ----
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 1.5rem 0 0.5rem 0;
    }
    .main-header h1 {
        font-size: 2.3rem;
        margin-bottom: 0.3rem;
        background: linear-gradient(90deg, #4A90D9, #7B61FF);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .subtitle {
        text-align: center;
        color: #666;
        font-size: 1rem;
        margin-bottom: 1.5rem;
    }
    .result-card {
        background: linear-gradient(135deg, #f8f9ff, #eef1ff);
        border-radius: 16px;
        padding: 1.5rem;
        margin-top: 1rem;
        border: 1px solid #e0e4f5;
    }
    .risk-badge {
        display: inline-block;
        padding: 0.25rem 0.9rem;
        border-radius: 20px;
        color: white;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .disclaimer-box {
        background: #FFF8E1;
        border-left: 4px solid #FFA500;
        border-radius: 8px;
        padding: 1rem 1.2rem;
        margin-top: 1.5rem;
        font-size: 0.9rem;
    }
    .stProgress > div > div {
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

# ---- Header ----
st.markdown("""
<div class="main-header">
    <h1>🩺 AI Skin Disease Detection</h1>
</div>
<p class="subtitle">
    EfficientNetB0 · Transfer Learning · HAM10000 Dataset · 7 Lesion Classes
</p>
""", unsafe_allow_html=True)

st.markdown(
    "Upload a clear, well-lit image of a skin lesion to get an **AI-assisted preliminary screening**. "
    "This is a portfolio/learning project — **not** a substitute for professional medical diagnosis."
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
            <h3 style="margin-top:0;">{info['name']}</h3>
            <span class="risk-badge" style="background-color:{info['risk_color']};">
                {info['risk']} Risk
            </span>
            <p style="margin-top:1rem; color:#444;">{info['description']}</p>
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
