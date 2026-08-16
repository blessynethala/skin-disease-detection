# 🩺 SkinSense AI

### An Intelligent Skin Disease Detection System Using Deep Learning

SkinSense AI is a deep learning-based web application that detects and classifies "7 types of skin lesions" from images using a fine-tuned "EfficientNetB0" model trained on the "HAM10000" dataset. Users can upload a skin lesion image and receive an AI-assisted preliminary screening — including the predicted condition, a confidence score, risk level, and relevant precautions.

> ⚠️ "Disclaimer:" This project is built for educational/portfolio purposes only. It is "not" a substitute for professional medical diagnosis. Always consult a certified dermatologist for accurate diagnosis and treatment.

---

## 🔗 Live Demo

Try SkinSense AI live →  https://skin-disease-detection-ilszz3msvhj3hvwpsqqyzl.streamlit.app/


---

## 📸 Preview

<img width="1913" height="777" alt="Screenshot 2026-08-16 203206" src="https://github.com/user-attachments/assets/0074d436-9b72-4ca3-ae4e-cb6f32dd79de" />

---

## 🧠 How It Works

1. User uploads a skin lesion image (JPG/PNG).
2. The image is preprocessed and resized to 224×224 to match the model's input.
3. A fine-tuned "EfficientNetB0" CNN predicts probabilities across 7 lesion classes.
4. The app displays the top prediction, confidence score, risk level, a short description, and relevant precautions.

---

## 🩹 Disease Classes

| Class | Condition | Type | Key Visual Traits |
|-------|-----------|------|--------------------|
| `mel` | Melanoma | Cancerous | Asymmetric shape, irregular borders, mixed colors |
| `nv` | Melanocytic Nevi (Common Mole) | Benign | Symmetric, uniform color, regular border |
| `bcc` | Basal Cell Carcinoma | Cancerous | Pink/pearly appearance, visible vessels |
| `akiec` | Actinic Keratoses | Pre-cancerous | Rough, scaly, red/pink patches |
| `bkl` | Benign Keratosis | Benign | "Stuck-on" waxy appearance, well-defined edges |
| `df` | Dermatofibroma | Benign | Small, firm nodule, often with central scar-like area |
| `vasc` | Vascular Lesions | Benign | Red/purple, blood-vessel related |

---

## 🛠️ Tech Stack

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| Google Colab | Model training (GPU-accelerated) |
| TensorFlow & Keras | Deep learning framework |
| EfficientNetB0 | Base CNN architecture (Transfer Learning) |
| OpenCV / PIL | Image preprocessing |
| Streamlit | Web app interface |
| GitHub | Version control & code hosting |
| Streamlit Community Cloud | Deployment |

---

## 📊 Dataset

"[HAM10000 ("Human Against Machine with 10000 training images")](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000)"

- 10,015 dermatoscopic images across 7 diagnostic categories
- Significant class imbalance (e.g., `nv`: 6,705 images vs `df`: 115 images), handled using computed class weights during training
- Split: 70% train / 15% validation / 15% test (stratified by class)

---

## 🏗️ Model & Training

- "Architecture:" EfficientNetB0 (pretrained on ImageNet) + GlobalAveragePooling2D + Dense(256, ReLU) + Dropout(0.3) + Dense(7, Softmax)
- **Training strategy:** Two-phase transfer learning
  - **Phase 1:** Base model frozen, only top layers trained (8 epochs)
  - **Phase 2:** Last 40 layers of the base model unfrozen and fine-tuned at a low learning rate (1e-5), for 20 epochs
- **Class imbalance handling:** Computed `class_weight` (balanced) applied during training
- **Data augmentation:** Random horizontal/vertical flips, brightness, and contrast adjustments (training set only)
- **Callbacks:** EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

---

## 📈 Results

| Metric | Score |
|---|---|
| **Test Accuracy** | 70.4% |
| **Test Precision** | 76.3% |
| **Test Recall** | 61.6% |

### Per-Class Performance

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| akiec | 0.44 | 0.71 | 0.54 | 49 |
| bcc | 0.52 | 0.73 | 0.61 | 77 |
| bkl | 0.48 | 0.58 | 0.52 | 165 |
| df | 0.25 | 0.59 | 0.35 | 17 |
| mel | 0.36 | 0.57 | 0.44 | 167 |
| nv | 0.97 | 0.74 | 0.84 | 1006 |
| vasc | 0.51 | 0.91 | 0.66 | 22 |

**Note:** Performance varies across classes, largely due to dataset imbalance and visual overlap between certain conditions (e.g., melanoma vs. other pigmented lesions) — a challenge even for human dermatologists without biopsy confirmation. This is an active limitation acknowledged in the app's disclaimer.

---

## ✨ Features

- 📤 Simple image upload interface
- 🔬 Real-time AI-assisted prediction
- 📊 Confidence scores across all 7 classes
- 🎨 Color-coded risk levels (Low / Moderate / High / Critical)
- 🛡️ Condition-specific precautions and recommendations
- ⚠️ Clear medical disclaimer throughout

---

## 🚀 Running Locally

```bash
# Clone the repository
git clone https://github.com/your-username/skin-disease-detection.git
cd skin-disease-detection

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run streamlit_app.py
```

---

## 📁 Project Structure

```
skin-disease-detection/
├── streamlit_app.py                # Main Streamlit application
├── final_skin_disease_model.keras  # Trained EfficientNetB0 model
├── background.png                  # UI background image
├── requirements.txt                # Python dependencies
├── README.md
└── LICENSE
```

---

## ⚠️ Disclaimer

SkinSense AI is an **AI-assisted preliminary screening tool** built as a learning/portfolio project. It is **not** a certified medical device and should **never** replace professional medical advice, diagnosis, or treatment. If you have any concerns about a skin lesion, please consult a certified dermatologist.

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
