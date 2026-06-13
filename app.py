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


# ---------------- Model Download ----------------
MODEL_PATH = "best_plant_disease_model.pth"
FILE_ID = "1RUurlqVd2nF7FdgjDrfKfv7butpeUaNo"

if not os.path.exists(MODEL_PATH):
    url = f"https://drive.google.com/uc?id={FILE_ID}"
    gdown.download(url, MODEL_PATH, quiet=False)


# ---------------- Load Class Names ----------------
with open("class_names.json", "r") as f:
    class_names = json.load(f)


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


# ---------------- Image Transform ----------------
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
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image)
        probs = torch.softmax(output, dim=1)
        conf, pred = torch.max(probs, dim=1)

    return class_names[pred.item()], conf.item()


# ---------------- Streamlit UI ----------------

# Page config
st.set_page_config(
    page_title="Plant Disease Classification",
    page_icon="🌿",
    layout="centered"
)

st.markdown("""
<style>
[data-testid="stFileUploader"] {
    border: 2px dashed #4CAF50;
    border-radius: 12px;
    padding: 12px;
    background-color: #1e1e1e;
}
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
# 🌿 Plant Disease Classifier
Detect plant diseases instantly using deep learning.
""")

st.markdown("<br>", unsafe_allow_html=True)

# Sidebar
st.sidebar.title("📌 About")
st.sidebar.info("""
This application uses an **Ensemble Deep Learning Model**:

- ResNet50
- ShuffleNetV2

Dataset:
- New Plant Diseases Dataset (Kaggle)
- 14 Unique Plants
- 38 Disease Classes
- 70,295 Training Images
- 17,572 Testing Images

Built with:
- PyTorch
- Streamlit
""")

# Supported plants
st.subheader("🌱 Supported Plants")

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
# Upload section
st.subheader("📤 Upload Leaf Image")
st.caption("Drag and drop your leaf image here or browse files.")

uploaded_file = st.file_uploader(
    "Drop image here",
    type=["jpg", "jpeg", "png"],
    label_visibility="collapsed"
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Uploaded Leaf Image",
        use_container_width=True
    )

    if st.button("🔍 Predict Disease"):
        with st.spinner("Analyzing leaf image..."):
            label, confidence = predict(image)

        st.subheader("📊 Prediction Result")
        st.metric("Disease", label)
        st.metric("Confidence", f"{confidence * 100:.2f}%")

        if confidence >= 0.90:
            st.success("High confidence prediction")
        elif confidence >= 0.70:
            st.warning("Medium confidence prediction")
        else:
            st.error("Low confidence prediction. Try a clearer image.")

# Footer
st.markdown(
    "<p style='text-align:center; margin-top:20px; color:gray;'>"
    "Built by Sayan Ghorai | Plant Disease Detection using Deep Learning"
    "</p>",
    unsafe_allow_html=True
)