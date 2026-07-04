import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import (
    resnet50,
    shufflenet_v2_x1_0
)
from PIL import Image
import json
import os
import gdown
import cv2
import numpy as np
import matplotlib.pyplot as plt

from gradcam import GradCAM, create_heatmap


# ---------------- Model Download ----------------
MODEL_PATH = "best_plant_disease_model.pth"
FILE_ID = "1RUurlqVd2nF7FdgjDrfKfv7butpeUaNo"

if not os.path.exists(MODEL_PATH):
    url = f"https://drive.google.com/uc?id={FILE_ID}"
    gdown.download(url, MODEL_PATH, quiet=False)


# ---------------- Load Files ----------------
with open("class_names.json", "r") as f:
    class_names = json.load(f)

with open("disease_info.json", "r") as f:
    disease_info = json.load(f)


# ---------------- Leaf Validation ----------------
def is_leaf_image(image, threshold=0.08):
    img = np.array(image)

    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    lower_green = np.array([15, 20, 20])
    upper_green = np.array([120, 255, 255])

    mask = cv2.inRange(hsv, lower_green, upper_green)

    green_ratio = np.count_nonzero(mask) / mask.size

    return green_ratio > threshold


# ---------------- Model ----------------
class ShuffleNetResNet50Ensemble(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.resnet = resnet50(weights=None)
        self.resnet.fc = nn.Identity()

        self.shufflenet = shufflenet_v2_x1_0(weights=None)
        self.shufflenet.fc = nn.Identity()

        self.fc = nn.Sequential(
            nn.Linear(2048 + 1024, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        r = self.resnet(x)
        s = self.shufflenet(x)
        x = torch.cat((r, s), dim=1)
        return self.fc(x)


# ---------------- Device ----------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------- Load Model ----------------
@st.cache_resource
def load_model():
    model = ShuffleNetResNet50Ensemble(len(class_names))

    model.load_state_dict(
        torch.load(
            MODEL_PATH,
            map_location=device
        )
    )

    model.to(device)
    model.eval()

    return model


model = load_model()


# ---------------- Transform ----------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ---------------- Prediction ----------------
def predict(image):
    image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image_tensor)
        probs = torch.softmax(output, dim=1)
        conf, pred = torch.max(probs, dim=1)

    label = class_names[pred.item()]

    return label, conf.item(), pred.item(), image_tensor


# ---------------- UI ----------------
st.set_page_config(
    page_title="Plant Disease Classification",
    page_icon="🌿",
    layout="wide"
)


# ---------------- Sidebar ----------------
st.sidebar.markdown("""
## 🌿 About the Project

AI-powered plant disease detection with explainable AI and treatment recommendations.

### Dataset Used

**New Plant Diseases Dataset (Kaggle)**

- 70k+ training images  
- 17k+ testing images  
- 14 plant species  
- 38 disease classes  

### Features

✅ Leaf Validation  
✅ Disease Classification  
✅ Grad-CAM Explainability  
✅ Treatment Suggestions  
✅ Prevention Recommendations  

---
""")

st.sidebar.success(
    "Smart agriculture for early disease detection."
)


# ---------------- Header ----------------
st.title("🌿 Plant Disease Classifier")
st.write("Detect plant diseases instantly using deep learning.")


# ---------------- Supported Plants ----------------
st.subheader("Supported Plants")

plants = [
    "🍎 Apple", "🫐 Blueberry", "🍒 Cherry", "🌽 Corn",
    "🍇 Grape", "🍊 Orange", "🫘 Soybean", "🎃 Squash",
    "🥔 Potato", "🍓 Strawberry", "🍑 Peach", "🌶 Pepper",
    "🍅 Tomato", "🍇 Raspberry"
]

st.markdown(
    "<div style='background-color:#17324d; padding:12px; border-radius:10px; font-size:18px;'>"
    + " | ".join(plants) +
    "</div>",
    unsafe_allow_html=True
)


# ---------------- Upload ----------------
st.subheader("📤 Upload Leaf Image")
st.caption("Drag and drop your leaf image here or browse files.")

uploaded_file = st.file_uploader(
    "Upload Leaf Image",
    type=["jpg", "jpeg", "png"]
)


if uploaded_file:

    original_image = Image.open(uploaded_file).convert("RGB")

    # Validate leaf
    if not is_leaf_image(original_image):
        st.error("❌ Please upload a valid leaf image.")
        st.stop()

    # Show only uploaded image
    st.image(
        original_image,
        caption="Uploaded Leaf",
        width=350
    )

    if st.button("🔍 Predict Disease"):

        label, confidence, pred_idx, image_tensor = predict(original_image)

        plant, disease = label.split("___")
        plant = plant.replace("_", " ")
        disease = disease.replace("_", " ")

        # Prediction
        st.subheader("📊 Prediction")

        col1, col2, col3 = st.columns(3)

        col1.metric("Plant", plant)
        col2.metric("Disease", disease)
        col3.metric("Confidence", f"{confidence * 100:.2f}%")

        # Healthy
        if "healthy" in label.lower():
            st.success("✅ Leaf appears healthy. No infected region detected.")

        else:
            target_layer = model.resnet.layer3[-1].conv3
            grad_cam = GradCAM(model, target_layer)

            # Generate CAM
            cam = grad_cam.generate(image_tensor, pred_idx)
            heatmap = create_heatmap(cam, original_image)

            # Keep same size as uploaded image
            h, w = original_image.size[1], original_image.size[0]

            fig, ax = plt.subplots(
                figsize=(w / 100, h / 100),
                dpi=100
            )

            fig.patch.set_facecolor("black")

            # Overlay original + heatmap
            ax.imshow(original_image)
            ax.imshow(
                heatmap,
                alpha=0.35,
                cmap="jet"
            )

            ax.axis("off")

            plt.tight_layout(pad=0)

            fig.canvas.draw()

            overlay = np.array(
                fig.canvas.renderer.buffer_rgba()
            )

            # Proper RGB conversion
            overlay_image = Image.fromarray(
                overlay
            ).convert("RGB")

            plt.close(fig)

            # Disease information
            info = disease_info.get(label, {})

            # Layout: Left = Grad-CAM | Right = Info
            col1, col2 = st.columns([1.2, 1])

            with col1:
                st.subheader("🔥 Grad-CAM Focus")
                st.image(
                    overlay_image,
                    caption="Detected Infection Area",
                    width=350
                )

            with col2:
                st.subheader("🦠 Cause")
                st.info(
                    info.get(
                        "cause",
                        "No information available."
                    )
                )

                st.subheader("💊 Treatment")
                st.success(
                    info.get(
                        "treatment",
                        "No information available."
                    )
                )

                st.subheader("🛡 Prevention")
                st.warning(
                    info.get(
                        "prevention",
                        "No information available."
                    )
                )

# ---------------- Footer ----------------
st.markdown("---")
st.caption(
    "Built by Sayan Ghorai | Plant Disease Detection using Deep Learning"
)