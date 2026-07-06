import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image

# Configuración de la página
st.set_page_config(page_title="Clasificador de Frutas IA", page_icon="🍎", layout="centered")

st.title("🍎 Clasificador de Frutas con Inteligencia Artificial")
st.write("Sube la foto de una fruta y el modelo entrenado con Fruits-360 identificará qué variedad es.")

# 1. Cargar las clases (etiquetas)
@st.cache_resource
def cargar_clases():
    with open("clases_frutas.txt", "read") as f:
        clases = [line.strip() for line in f.readlines()]
    return clases

try:
    class_names = cargar_clases()
except FileNotFoundError:
    st.error("No se encontró el archivo 'clases_frutas.txt'. Asegúrate de subirlo a tu repositorio.")
    st.stop()

# 2. Cargar el modelo entrenado
@st.cache_resource
def cargar_modelo():
    return tf.keras.models.load_model("modelo_frutas.keras")

try:
    model = cargar_modelo()
except Exception as e:
    st.error(f"Error al cargar el modelo: {e}")
    st.stop()

# 3. Interfaz de carga de imágenes
uploaded_file = st.file_uploader("Elige una imagen de fruta...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Mostrar la imagen seleccionada por el usuario
    image = Image.open(uploaded_file)
    st.image(image, caption="Imagen subida", use_container_width=True)
    
    st.write("---")
    st.write("🔄 Procesando y clasificando...")
    
    # Preprocesamiento idéntico al entrenamiento
    # Convertimos a RGB por si la imagen tiene canal Alfa (PNG)
    image_rgb = image.convert("RGB")
    image_resized = image_rgb.resize((100, 100))  # Tamaño de Fruits-360
    img_array = np.array(image_resized) / 255.0   # Normalización
    img_array = np.expand_dims(img_array, axis=0)  # Añadir dimensión de Batch (1, 100, 100, 3)
    
    # Realizar la predicción
    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])  # Convertir a probabilidades reales
    
    id_clase_predicha = np.argmax(predictions[0])
    clase_predicha = class_names[id_clase_predicha]
    confianza = np.max(predictions[0]) * 100  # Porcentaje de acierto del softmax
    
    # Mostrar resultados estilizados
    st.success(f"### Predicción: **{clase_predicha}**")
    st.metric(label="Nivel de Confianza", value=f"{confianza:.2f}%")
