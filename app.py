import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Configurar la página de Streamlit
st.set_page_config(page_title="Predicción de Aprobación de Curso", layout="centered")

st.title("Predicción de Nota Final - Curso")
st.write("Introduce los datos del estudiante para estimar su nota final utilizando el modelo optimizado de Bagging.")

# 1. Cargar artefactos necesarios de forma segura
@st.cache_resource
def load_artifacts():
    try:
        columnas_one_hot = joblib.load('/content/one_hot_columns.joblib')
        scaler = joblib.load('/content/min_max_scaler.joblib')
        model = joblib.load('/content/bagging_optimizado.joblib')
        return columnas_one_hot, scaler, model
    except Exception as e:
        st.error(f"Error al cargar los archivos .joblib: {e}")
        return None, None, None

columnas_one_hot, scaler, model = load_artifacts()

if columnas_one_hot and scaler and model:
    # 2. Formulario de entrada de usuario
    st.header("Datos del Estudiante")
    
    # Extraer las categorías posibles para la variable 'Felder'
    # Basado en la lista: ['Felder_equilibrio', 'Felder_intuitivo', 'Felder_reflexivo', 'Felder_secuencial', 'Felder_sensorial', 'Felder_verbal', 'Felder_visual']
    categorias_felder = [col.replace('Felder_', '') for col in columnas_one_hot if col.startswith('Felder_')]
    
    felder_selected = st.selectbox("Estilo de Aprendizaje (Felder)", opciones=categorias_felder)
    examen_admision = st.slider("Nota de Examen de Admisión", min_value=0.0, max_value=5.0, value=3.8, step=0.05)
    
    if st.button("Calcular Predicción"):
        # 3. Procesar datos de entrada exactamente igual que el flujo anterior
        df_input = pd.DataFrame([{'Felder': felder_selected, 'Examen_admisión': examen_admision}])
        
        # Aplicar codificación One-Hot manual de acuerdo a la lista cargada
        for col in columnas_one_hot:
            if col.startswith('Felder_'):
                categoria = col.replace('Felder_', '')
                df_input[col] = 1.0 if felder_selected == categoria else 0.0
        
        # Aplicar el Min-Max Scaler cargado
        df_input['Examen_admision_scaled'] = scaler.transform(df_input[['Examen_admisión']])[0][0]
        
        # Seleccionar y ordenar las columnas según las que espera el modelo
        columnas_finales = [col for col in columnas_one_hot if col in df_input.columns]
        df_procesado = df_input[columnas_finales]
        
        # 4. Realizar la predicción con el modelo
        prediccion = model.predict(df_procesado)[0]
        
        # Mostrar resultados en pantalla
        st.success(f"### Nota Final Estimada: {prediccion:.3f}")
        
        # Mostrar detalle de variables enviadas al modelo
        with st.expander("Ver variables procesadas enviadas al modelo"):
            st.dataframe(df_procesado)
else:
    st.warning("Por favor, asegúrate de que los archivos 'one_hot_columns.joblib', 'min_max_scaler.joblib' y 'bagging_optimizado.joblib' se encuentren en la ruta correcta.")
