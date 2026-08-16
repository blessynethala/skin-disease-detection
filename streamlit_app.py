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
        'risk': 'Moderate - Pre-cancerous, consult a dermatologist'
    },
    'bcc': {
        'name': 'Basal Cell Carcinoma',
        'description': 'The most common type of skin cancer. Grows slowly and rarely spreads, but needs treatment.',
        'risk': 'High - Cancerous, needs medical attention'
    },
    'bkl': {
        'name': 'Benign Keratosis',
        'description': 'A non-cancerous skin growth, often appearing as a waxy or scaly raised patch.',
        'risk': 'Low - Benign, but monitor for changes'
    },
    'df': {
        'name': 'Dermatofibroma',
        'description': 'A common benign skin nodule, usually harmless, often caused by minor skin injury.',
        'risk': 'Low - Benign'
    },
    'mel': {
        'name': 'Melanoma',
        'description': 'The most serious and dangerous type of skin cancer. Can spread quickly if not treated early.',
        'risk': 'Critical - Cancerous, seek immediate medical consultation'
    },
    'nv': {
        'name': 'Melanocytic Nevi (Common Mole)',
        'description': 'A common, usually harmless mole made of pigment-producing cells.',
        'risk': 'Low - Benign, but monitor for changes in size/color'
    },
    'vasc': {
        'name': 'Vascular Lesions',
        'description': 'Skin marks caused by blood vessel abnormalities, such as birthmarks or angiomas.',
        'risk': 'Low - Benign'
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

st.title("🩺 AI Skin Disease Detection")
st.write(
    "Upload a clear, well-lit image of a skin lesion to get an AI-assisted preliminary "
    "screening. This tool uses EfficientNetB0 (transfer learning) trained on the HAM10000 "
    "dataset (7 skin lesion classes)."
)
st.info("This is a portfolio/learning project — NOT a substitute for professional medical diagnosis.")

model = load_trained_model()

uploaded_file = st.file_uploader("Upload Skin Lesion Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    with st.spinner("Analyzing image..."):
        preds = predict(image, model)

    top_idx = int(np.argmax(preds))
    top_class = CLASSES[top_idx]
    top_conf = float(preds[top_idx]) * 100
    info = DISEASE_INFO[top_class]

    st.subheader(f"Predicted: {info['name']}")
    st.metric("Confidence", f"{top_conf:.1f}%")
    st.write(f"**Risk Level:** {info['risk']}")
    st.write(f"**Description:** {info['description']}")

    st.write("### Confidence Scores (All Classes)")
    sorted_indices = np.argsort(preds)[::-1]
    for idx in sorted_indices:
        st.write(f"{CLASSES[idx]}: {preds[idx]*100:.1f}%")
        st.progress(float(preds[idx]))

    st.warning(
        "⚠️ Disclaimer: This is an AI-assisted preliminary screening tool, NOT a medical "
        "diagnosis. Please consult a certified dermatologist for accurate diagnosis and treatment. "
        "This model achieves ~70% overall test accuracy; performance varies by disease type."
    )
