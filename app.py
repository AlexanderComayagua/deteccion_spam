import streamlit as st
import joblib
import json
import os
import re

# Configuración de la página
st.set_page_config(
    page_title="Detector de Spam | Árbol de Decisión",
    page_icon="🛡️",
    layout="centered"
)

# Cargar metadatos y modelo
@st.cache_resource
def cargar_recursos():
    ruta_modelo = "modelo_spam_arbol.pkl"
    ruta_meta = "model_metadata.json"
    
    modelo = None
    metadata = {}
    
    if os.path.exists(ruta_modelo):
        modelo = joblib.load(ruta_modelo)
    if os.path.exists(ruta_meta):
        with open(ruta_meta, "r", encoding="utf-8") as f:
            metadata = json.load(f)
            
    return modelo, metadata

modelo, metadata = cargar_recursos()

# Encabezado principal
st.title("🛡️ Detector de Correos Spam")
st.markdown(
    "Clasificación de mensajes mediante **Procesamiento de Lenguaje Natural (Bag of Words)** "
    "y un **Árbol de Decisión**."
    "Hecho por Cristpher Flores y Angel Zavala"
)

# Verificar que el modelo exista
if modelo is None:
    st.error("⚠️ No se encontró el archivo `modelo_spam_arbol.pkl` en el directorio raíz.")
    st.stop()

# Ejemplos rápidos para probar
st.subheader("✉️ Ingresa el contenido del correo o mensaje")

mensaje_usuario = st.text_area(
    "Texto del mensaje (preferiblemente en inglés, idioma de entrenamiento):",
    value=texto_inicial,
    height=150,
    placeholder="Escribe o pega aquí el contenido del correo electrónico..."
)

if st.button("🔍 Analizar Mensaje", type="primary", use_container_width=True):
    if not mensaje_usuario.strip():
        st.warning("Por favor, ingresa un texto antes de analizar.")
    else:
        # Predicción y probabilidades
        prediccion = modelo.predict([mensaje_usuario])[0]
        probabilidades = modelo.predict_proba([mensaje_usuario])[0]
        
        prob_ham = probabilidades[0] * 100
        prob_spam = probabilidades[1] * 100
        
        st.divider()
        
        if prediccion == 1:
            st.error("🚨 **RESULTADO: POSIBLE SPAM DETECTADO**")
            st.write(f"El modelo clasifica este mensaje como **Spam** con una confianza del **{prob_spam:.2f}%**.")
        else:
            st.success("✅ **RESULTADO: MENSAJE LEGÍTIMO (HAM)**")
            st.write(f"El modelo clasifica este mensaje como **Legítimo** con una confianza del **{prob_ham:.2f}%**.")
        
        # Desglose de probabilidades
        col1, col2 = st.columns(2)
        col1.metric("Probabilidad Legítimo (Ham)", f"{prob_ham:.2f}%")
        col2.metric("Probabilidad Spam", f"{prob_spam:.2f}%")
        
        # Palabras detectadas en el vocabulario del modelo
        vectorizer = modelo.named_steps.get("vectorizer")
        if vectorizer:
            analyzer = vectorizer.build_analyzer()
            tokens = analyzer(mensaje_usuario)
            vocab = vectorizer.vocabulary_
            palabras_conocidas = sorted(list(set([t for t in tokens if t in vocab])))
            
            with st.expander("🔎 Ver palabras clave reconocidas por el vocabulario del modelo"):
                if palabras_conocidas:
                    st.write(", ".join([f"`{p}`" for p in palabras_conocidas]))
                else:
                    st.caption("Ninguna palabra del mensaje coincide con las 3,000 palabras principales del vocabulario.")
