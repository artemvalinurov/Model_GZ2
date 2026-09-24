import os
import sys
import codecs
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
import streamlit as st
import streamlit.components.v1 as components
import gdown

# Намертво привязываем пути поиска соседних модулей к текущей папке
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from model import GalaxyNet

# 1. МОБИЛЬНАЯ НАСТРОЙКА ЭКРАНА (wide-режим на весь дисплей телефона)
st.set_page_config(
    page_title="Galaxy Classifier ResNet18",
    page_icon="🌌",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# 2. КЭШИРОВАННАЯ ЗАГРУЗКА И АВТОМАТИЧЕСКОЕ СКАЧИВАНИЕ ВЕСОВ С МОДЕЛИ (162 МБ)
@st.cache_resource
def load_galaxy_model():
    device = torch.device("cpu") # Сервер хостинга работает строго на процессоре
    model = GalaxyNet()
    weights_path = "gz2_best (1).pt"
    
    # Если файла весов физически нет в репозитории — качаем его из вашего облака
    if not os.path.exists(weights_path):
        with st.spinner("🚀 Загрузка весов модели ResNet-18 (162 МБ) из облака... Пожалуйста, подождите."):
            # 🌟 ВСТАВЬТЕ СЮДА ВАШУ ПРЯМУЮ ССЫЛКУ СКАЧИВАНИЯ ИЗ ЯНДЕКС/GOOGLE ДИСКА
            url = "https://your-direct-cloud-link-here.pt" 
            gdown.download(url, weights_path, quiet=False)
            
    if os.path.exists(weights_path):
        ckpt = torch.load(weights_path, map_location=device)
        # Поддерживаем как чистый словарь весов, так и полный чекпоинт обучения из Kaggle
        state_dict = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
        model.load_state_dict(state_dict)
        model.to(device).eval()
        return model, device, weights_path
    return None, device, None

model, device, used_path = load_galaxy_model()

# 3. ТОЧНЫЙ КОНВЕЙЕР ОБРАБОТКИ СНИМКОВ (Копия val_tf из Kaggle)
eval_transforms = transforms.Compose([
    transforms.CenterCrop(300),     # Вырезаем центральные 300x300 пикселей (ядро галактики)
    transforms.Resize((224, 224)),  # Сжимаем до 224x224 под вход ResNet/ViT
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ============================================================
# 4. АДАПТИВНЫЙ ИНТЕРФЕЙС ВЕБ-САЙТА (ДЛЯ ПК И СМАРТФОНОВ)
# ============================================================
st.title("🌌 Многоголовый анализатор галактик (ResNet-18)")
st.write("Профессиональный инференс на базе архитектуры GalaxyNet по стандартам обзора Galaxy Zoo 2.")

if model is not None:
    st.success("🤖 Нейросеть ResNet-18 успешно инициализирована и готова к работе.")
else:
    st.error("❌ Критическая ошибка: Не удалось загрузить веса модели `gz2_best (1).pt`!")

uploaded_file = st.file_uploader("Загрузите снимок объекта (JPG, JPEG, PNG)...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None and model is not None:
    image = Image.open(uploaded_file).convert("RGB")
    
    # На ПК элементы встанут в два столбца, на телефонах — ровно друг под друга
    col1, col2 = st.columns(2)
    
    with col1:
        st.image(image, caption="Входное изображение", use_container_width=True)
        
    with col2:
        st.subheader("Морфологический профиль модели:")
        
        # Подготовка тензора
        input_tensor = eval_transforms(image).unsqueeze(0).to(device)
        
        # Инференс по всем головам без подсчета градиентов
        with torch.no_grad():
            out = model(input_tensor)
            
            # Применяем правильные математические функции активации из вашего блокнота
            p_type = F.softmax(out["type"].float(), dim=-1).cpu().squeeze(0).numpy() # [smooth, disk, star]
            p_odd  = F.softmax(out["odd"].float(), dim=-1).cpu().squeeze(0).numpy()  # [normal, merger, disturbed, other]
            p_edge = torch.sigmoid(out["edge"].float()).cpu().item()                 # Вероятность ребра
            p_bar  = torch.sigmoid(out["bar"].float()).cpu().item()                  # Вероятность бара
            
        # --- Вывод результатов Головы 1 (Базовый тип) ---
        st.write(f"**Гладкая / Эллиптическая:** {p_type[0]*100:.1f}%")
        st.progress(float(p_type[0]))
        
        st.write(f"**Дисковая / Спиральная:** {p_type[1]*100:.1f}%")
        st.progress(float(p_type[1]))
        
        st.write(f"**Звезда / Артефакт кадра:** {p_type[2]*100:.1f}%")
        st.progress(float(p_type[2]))
        
        st.write("---")
        
        # --- Вывод результатов созависимых Голов (Ребро и Бар) ---
        st.write(f"**Ракурс диска (С ребра):** {p_edge*100:.1f}%")
        st.progress(float(p_edge))
        
        st.write(f"**Центральное ядро (Наличие Бара):** {p_bar*100:.1f}%")
        st.progress(float(p_bar))
        
        st.write(f"**Динамика слияния (Merger):** {p_odd[1]*100:.1f}%")
        st.progress(float(p_odd[1]))

        # --- СЕКЦИЯ ЭКСПЕРТНОГО ТЕКСТОВОГО ВЕРДИКТА ---
        st.subheader("Вердикт нейросети:")
        
        type_idx = np.argmax(p_type)
        if type_idx == 2:
            st.warning("⚠️ Внимание: ИИ определил объект как яркую звезду переднего плана или дефект матрицы телескопа.")
        elif type_idx == 0:
            st.info(f"Объект классифицирован как **Гладкая (Эллиптическая)** галактика с уверенностью {p_type[0]*100:.1f}%.")
        else:
            # Описываем дисковую структуру
            форма = "Спиральная"
            ракурс = "видна строго с ребра (Edge-on)" if p_edge >= 0.5 else "развернута плашмя (Face-on)"
            бар = "с выраженной перемычкой (баром)" if p_bar >= 0.5 else "без центрального бара"
            
            odd_idx = np.argmax(p_odd)
            особенность = "стабильная одиночная система"
            if odd_idx == 1:
                особенность = "активная стадия слияния (Merger)"
            elif odd_idx == 2:
                особенность = "гравитационно возмущенная структура (Disturbed)"
                
            st.info(f"Объект классифицирован как **{форма}** галактика ({p_type[1]*100:.1f}%). Система **{ракурс}**, сформирована **{бар}**. Динамическое состояние: **{особенность}**.")

# ============================================================
# 5. АВТОМАТИЧЕСКАЯ ОТРИСОВКА ВИЛКИ ХАББЛА (ИЗ ВНЕШНИХ ФАЙЛОВ)
# ============================================================
st.write("---")
st.subheader("🗺️ Справочник морфологии: Карта Эдвина Хаббла")

current_dir = os.path.dirname(os.path.abspath(__file__))
css_path = os.path.join(current_dir, "style.css")
html_path = os.path.join(current_dir, "1.html")

if os.path.exists(html_path) and os.path.exists(css_path):
    with codecs.open(css_path, "r", "utf-8") as f:
        css_content = f.read()
    with codecs.open(html_path, "r", "utf-8") as f:
        html_content = f.read()
        
    full_html_code = f"<style>{css_content}</style>{html_content}"
    # Отрисовываем изолированный адаптивный iframe блок
    components.html(full_html_code, height=320, scrolling=False)
else:
    st.warning("⚠️ Файлы `1.html` или `style.css` не найдены в корневой папке репозитория!")
