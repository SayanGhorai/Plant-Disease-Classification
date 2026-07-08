# 🌿 Plant Disease Classification with Explainable AI

An AI-powered deep learning web application for plant disease detection using leaf images. The project combines an ensemble of **ResNet50** and **ShuffleNetV2** with **Grad-CAM Explainable AI** to classify plant diseases and visualize the model's decision-making process.

## 🚀 Live Demo

**Streamlit App:**  
https://plant-disease-classification-4wlqmvvq82hdzlwqgdpqsz.streamlit.app/

---

## 📌 Project Overview

Plant diseases can significantly reduce agricultural productivity and crop yield. Early disease detection enables farmers to take timely action and minimize losses.

This project uses deep learning to:

- Detect plant diseases from leaf images
- Provide confidence scores for predictions
- Visualize model attention using Grad-CAM
- Suggest disease causes, treatments, and prevention methods
- Validate uploaded images to ensure they contain plant leaves

Users simply upload a leaf image and receive an instant diagnosis.

---

## 🧠 Model Architecture

This project uses an **Ensemble Deep Learning Architecture**:

- **ResNet50** for deep feature extraction
- **ShuffleNetV2** for lightweight feature extraction
- Feature concatenation from both networks
- Fully connected layers for final classification

### Architecture Flow

```text
Leaf Image
    ↓
ResNet50 + ShuffleNetV2
    ↓
Feature Concatenation
    ↓
Dense Layers
    ↓
Disease Prediction
    ↓
Grad-CAM Visualization
```

---

## 🔍 Explainable AI

The application integrates **Grad-CAM (Gradient-weighted Class Activation Mapping)** to improve model interpretability.

Grad-CAM helps:

- Visualize regions influencing predictions
- Increase user trust in model decisions
- Understand model behavior
- Detect potential dataset bias

---

## ✨ Features

✅ Plant Disease Classification  
✅ Grad-CAM Explainability  
✅ Leaf Image Validation  
✅ Confidence Score Prediction  
✅ Disease Cause Identification  
✅ Treatment Recommendations  
✅ Prevention Suggestions  
✅ Streamlit Web Application  
✅ Automatic Model Download using Google Drive

---

## 📂 Dataset

Dataset used:

**New Plant Diseases Dataset (Kaggle)**

Dataset Statistics:

- **14 Plant Species**
- **38 Disease Classes**
- **70,295 Training Images**
- **17,572 Validation/Test Images**

### Supported Plants

- 🍎 Apple
- 🫐 Blueberry
- 🍒 Cherry
- 🌽 Corn
- 🍇 Grape
- 🍊 Orange
- 🫘 Soybean
- 🎃 Squash
- 🥔 Potato
- 🍓 Strawberry
- 🍑 Peach
- 🌶 Pepper
- 🍅 Tomato
- 🍇 Raspberry

---

## ⚙️ Training Details

- Image Size: **224 × 224**
- Batch Size: **32**
- Optimizer: **Adam**
- Loss Function: **CrossEntropyLoss**
- Early Stopping Applied
- Best Model Checkpoint Saved Automatically

### Performance

| Metric              | Score  |
| ------------------- | ------ |
| Validation Accuracy | 99.13% |
| Test Accuracy       | 99.11% |

---

## 🖥️ Application Preview

### Homepage

![Homepage](assets/homepage.png)

---

### Upload Leaf Image

![Upload Leaf](assets/upload_leaf.png)

---

### Diseased Leaf Prediction

![Disease Prediction](assets/prediction_disease.png)

---

### Healthy Leaf Prediction

![Healthy Prediction](assets/prediction_healthy.png)

---

## 📁 Project Structure

```text
Plant-Disease-Classification/
│
├── app.py
├── gradcam.py
├── disease_info.json
├── class_names.json
├── requirements.txt
├── README.md
│
├── assets/
│   ├── homepage.png
│   ├── upload_leaf.png
│   ├── prediction_disease.png
│   └── prediction_healthy.png
│
└── .gitignore
```

---

## 🛠 Installation

Clone the repository:

```bash
git clone https://github.com/SayanGhorai/Plant-Disease-Classification.git
cd Plant-Disease-Classification
```

Create virtual environment:

```bash
python -m venv venv
```

Activate environment:

### Windows

```bash
venv\Scripts\activate
```

### Linux/Mac

```bash
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
streamlit run app.py
```

---

## 📥 Model Loading

The trained model is hosted externally due to GitHub file size limitations.

The application automatically downloads the model from Google Drive using:

- **gdown**

No manual model download is required.

---

## ⚠ Limitations

- Real-world images may differ from training images.
- Background clutter can affect predictions.
- Grad-CAM may occasionally highlight contextual regions in addition to disease spots.
- Some diseases with subtle symptoms remain difficult to localize precisely.

---

## 🔮 Future Improvements

- Top-3 disease predictions
- Leaf segmentation for background removal
- Better real-world image robustness
- Lightweight mobile deployment
- Multi-stage disease diagnosis pipeline
- Disease severity estimation

---

## 🧰 Tech Stack

- Python
- PyTorch
- Torchvision
- Streamlit
- OpenCV
- PIL
- NumPy
- Matplotlib
- gdown

---

## 👨‍💻 Author

**Sayan Ghorai**  
M.Tech in Artificial Intelligence and Data Science

GitHub: https://github.com/SayanGhorai

LinkedIn: https://www.linkedin.com/in/sayan-ghorai-3a1202262/
