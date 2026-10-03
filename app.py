import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from io import BytesIO

st.set_page_config(page_title="Predicción de Nota Final", layout="centered")
st.title("Predicción de Nota Final - Curso")
st.write("Estima la nota final con el modelo optimizado de Bagging, de forma individual o cargando un archivo Excel.")

BASE_DIR = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()

COL_FELDER = "Felder"
COL_EXAMEN = "Examen_admisión"
COL_SCALED = "Examen_admision_scaled"


@st.cache_resource
def load_artifacts():
    try:
        columnas = joblib.load(os.path.join(BASE_DIR, 'one_hot_columns.joblib'))
        scaler = joblib.load(os.path.join(BASE_DIR, 'min_max_scaler.joblib'))
        model = joblib.load(os.path.join(BASE_DIR, 'bagging_optimizado.joblib'))
        return list(columnas), scaler, model
    except Exception as e:
        st.error(f"Error al cargar los archivos .joblib: {e}")
        return None, None, None


def preparar(df, columnas, scaler, model):
    """Convierte un DataFrame con Felder y Examen_admisión en la matriz que espera el modelo."""
    df = df.copy()
    df[COL_EXAMEN] = pd.to_numeric(df[COL_EXAMEN], errors="coerce")

    # One-Hot manual
    for col in columnas:
        if col.startswith('Felder_'):
            cat = col.replace('Felder_', '')
            df[col] = (df[COL_FELDER].astype(str).str.strip() == cat).astype(float)

    # Escalado
    df[COL_SCALED] = scaler.transform(df[[COL_EXAMEN]]).ravel()

    # Orden de columnas
    if hasattr(model, "feature_names_in_"):
        finales = list(model.feature_names_in_)
    else:
        finales = list(columnas)
        if COL_SCALED not in finales:
            finales.append(COL_SCALED)

    for c in finales:
        if c not in df.columns:
            df[c] = 0.0
    return df[finales]


def plantilla_excel(categorias):
    ejemplo = pd.DataFrame({
        COL_FELDER: categorias[:2] if len(categorias) >= 2 else categorias,
        COL_EXAMEN: [3.8, 4.2][:max(1, min(2, len(categorias)))]
    })
    buf = BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as w:
        ejemplo.to_excel(w, index=False, sheet_name="Estudiantes")
    return buf.getvalue()


columnas_one_hot, scaler, model = load_artifacts()

if columnas_one_hot is not None and scaler is not None and model is not None:
    categorias_felder = [c.replace('Felder_', '') for c in columnas_one_hot if c.startswith('Felder_')]

    tab1, tab2 = st.tabs(["Estudiante individual", "Cargar Excel"])

    # ---------- Individual ----------
    with tab1:
        st.header("Datos del Estudiante")
        felder_sel = st.selectbox("Estilo de Aprendizaje (Felder)", options=categorias_felder)
        examen = st.slider("Nota de Examen de Admisión", 0.0, 5.0, 3.8, 0.05)

        if st.button("Calcular Predicción"):
            df_in = pd.DataFrame([{COL_FELDER: felder_sel, COL_EXAMEN: examen}])
            X = preparar(df_in, columnas_one_hot, scaler, model)
            pred = model.predict(X)[0]
            st.success(f"### Nota Final Estimada: {pred:.3f}")
            with st.expander("Ver variables procesadas enviadas al modelo"):
                st.dataframe(X)

    # ---------- Excel ----------
    with tab2:
        st.header("Predicción masiva desde Excel")
        st.write(f"El archivo debe tener las columnas **{COL_FELDER}** y **{COL_EXAMEN}**.")
        st.caption(f"Valores válidos de Felder: {', '.join(categorias_felder)}")

        st.download_button(
            "Descargar plantilla de ejemplo",
            data=plantilla_excel(categorias_felder),
            file_name="plantilla_estudiantes.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

        archivo = st.file_uploader("Sube tu archivo Excel", type=["xlsx", "xls"])

        if archivo is not None:
            try:
                xls = pd.ExcelFile(archivo)
                hoja = st.selectbox("Hoja", xls.sheet_names) if len(xls.sheet_names) > 1 else xls.sheet_names[0]
                df_excel = pd.read_excel(xls, sheet_name=hoja)
                df_excel.columns = [str(c).strip() for c in df_excel.columns]

                faltantes = [c for c in (COL_FELDER, COL_EXAMEN) if c not in df_excel.columns]
                if faltantes:
                    st.error(f"Faltan columnas en el archivo: {', '.join(faltantes)}")
                    st.write("Columnas encontradas:", list(df_excel.columns))
                else:
                    st.write(f"Filas leídas: {len(df_excel)}")
                    st.dataframe(df_excel.head())

                    if st.button("Predecir todos"):
                        df_excel[COL_EXAMEN] = pd.to_numeric(df_excel[COL_EXAMEN], errors="coerce")
                        felder_ok = df_excel[COL_FELDER].astype(str).str.strip().isin(categorias_felder)
                        valido = df_excel[COL_EXAMEN].notna() & felder_ok

                        resultado = df_excel.copy()
                        resultado["Nota_Final_Estimada"] = np.nan
                        resultado["Observación"] = ""
                        resultado.loc[~valido, "Observación"] = "Dato inválido (Felder o examen)"

                        if valido.any():
                            X = preparar(df_excel[valido], columnas_one_hot, scaler, model)
                            resultado.loc[valido, "Nota_Final_Estimada"] = np.round(model.predict(X), 3)

                        n_inv = int((~valido).sum())
                        if n_inv:
                            st.warning(f"{n_inv} fila(s) con datos inválidos; quedaron sin predicción.")

                        st.success("Predicciones completadas")
                        st.dataframe(resultado)

                        buf = BytesIO()
                        with pd.ExcelWriter(buf, engine="openpyxl") as w:
                            resultado.to_excel(w, index=False, sheet_name="Predicciones")
                        st.download_button(
                            "Descargar resultados en Excel",
                            data=buf.getvalue(),
                            file_name="predicciones_notas.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        )
            except Exception as e:
                st.error(f"No se pudo procesar el archivo: {e}")
else:
    st.warning("Asegúrate de que 'one_hot_columns.joblib', 'min_max_scaler.joblib' y 'bagging_optimizado.joblib' estén en la misma carpeta que este script.")
