import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import (
    resnet50,
    shufflenet_v2_x1_0,
    ResNet50_Weights,
    ShuffleNet_V2_X1_0_Weights
)
from PIL import Image
import json


# ---------------- Load class names ----------------
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


# ---------------- Load trained model ----------------
@st.cache_resource
def load_model():
    model = ShuffleNetResNet50Ensemble(len(class_names))
    model.load_state_dict(
        torch.load(
            "best_plant_disease_model.pth",
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
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image)
        probs = torch.softmax(output, dim=1)
        conf, pred = torch.max(probs, dim=1)

    return class_names[pred.item()], conf.item()


# ---------------- UI ----------------
st.title("🌿 Plant Disease Classification")
st.write("Upload a leaf image to detect plant disease")

uploaded_file = st.file_uploader(
    "Upload image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Uploaded Image",
        use_container_width=True
    )

    if st.button("Predict"):
        label, confidence = predict(image)

        st.success(f"Prediction: {label}")
        st.info(f"Confidence: {confidence*100:.2f}%")