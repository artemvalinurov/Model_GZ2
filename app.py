import os
import sys
import codecs
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
import streamlit as st
import gdown

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from model import GalaxyNet

st.set_page_config(page_title="Galaxy Classifier ResNet18", page_icon="🌌", layout="wide")



@st.cache_resource
def load_galaxy_model():
    device = torch.device("cpu")
    model = GalaxyNet()
    weights_path = "gz2_best (1).pt"
    
    # 🌟 Если файла еще нет на сервере Streamlit — качаем его из облака автоматичеки
    if not os.path.exists(weights_path):
        with st.spinner("Загрузка тяжелых весов модели ResNet-18 из облачного хранилища..."):
            url = "https://drive.google.com/file/d/1c9j4upI5b33d5XUIe7X2qUM85HheKDuX/view?usp=sharing"
            gdown.download(url, weights_path, quiet=False)
            
    if os.path.exists(weights_path):
        ckpt = torch.load(weights_path, map_location=device)
        model.load_state_dict(ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt)
        model.to(device).eval()
        return model, device, weights_path
    return None, device, None


model, device, used_path = load_galaxy_model()
st.title("🌌 Анализатор глубокого космоса (ResNet-18)")

uploaded_file = st.file_uploader("Выберите снимок галактики...", type=["jpg", "jpeg", "png"])
if uploaded_file is not None and model is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Входной кадр", use_container_width=True)
