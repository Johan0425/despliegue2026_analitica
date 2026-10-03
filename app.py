import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from io import BytesIO

# ------------------ CONFIG Y ESTILO ------------------

st.set_page_config(
    page_title="Predicción de Nota Final",
    page_icon="🎓",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Estilo con blur, bordes redondeados y animaciones
st.markdown("""
<style>
/* Tarjetas / contenedores con blur */
.card {
    background: rgba(255, 255, 255, 0.65);
    backdrop-filter: blur(10px);
    -webkit-backdrop-filter: blur(10px);
    border-radius: 18px;
    padding: 20px 24px;
    box-shadow:
        0 10px 30px rgba(0, 0, 0, 0.06),
        0 2px 8px rgba(0, 0, 0, 0.04);
    border: 1px solid rgba(0, 0, 0, 0.06);
    margin-bottom: 24px;
    transition: transform 0.2s ease, box-shadow 0.2s ease, background 0.2s ease;
}
.card:hover {
    transform: translateY(-2px);
    box-shadow:
        0 18px 40px rgba(0, 0, 0, 0.09),
        0 4px 12px rgba(0, 0, 0, 0.06);
    background: rgba(255, 255, 255, 0.75);
}

/* Botones con animación sutil */
.stButton > button {
    border-radius: 12px;
    border: 1px solid rgba(0,0,0,0.08);
    font-weight: 500;
    box-shadow:
        0 8px 20px rgba(0, 0, 0, 0.06),
        0 2px 6px rgba(0, 0, 0, 0.04);
    transition: transform 0.15s ease, box-shadow 0.15s ease;
}
.stButton > button:hover {
    transform: translateY(-1px);
    box-shadow:
        0 14px 28px rgba(0, 0, 0, 0.08),
        0 4px 10px rgba(0, 0, 0, 0.05);
}

/* Métricas */
.metric-card {
    background: rgba(255, 255, 255, 0.6);
    backdrop-filter: blur(8px);
    -webkit-backdrop-filter: blur(8px);
    border-radius: 14px;
    padding: 14px 16px;
    box-shadow:
        0 8px 20px rgba(0, 0, 0, 0.05),
        0 2px 6px rgba(0, 0, 0, 0.03);
    border: 1px solid rgba(0,0,0,0.05);
    text-align: center;
    transition: transform 0.15s ease, box-shadow 0.15s ease, background 0.15s ease;
}
.metric-card:hover {
    transform: translateY(-1px);
    box-shadow:
        0 14px 26px rgba(0, 0, 0, 0.07),
        0 4px 10px rgba(0, 0, 0, 0.04);
    background: rgba(255, 255, 255, 0.72);
}
.metric-value {
    font-size: 1.6rem;
    font-weight: 700;
    letter-spacing: -0.02em;
}
.metric-label {
    font-size: 0.85rem;
    color: #6e6e73;
    margin-top: 4px;
}

/* Animaciones suaves de entrada */
@keyframes fadeInUp {
    from { opacity: 0; transform: translateY(10px); }
    to   { opacity: 1; transform: translateY(0); }
}
.fade-in {
    animation: fadeInUp 0.35s ease forwards;
}
</style>
""", unsafe_allow_html=True)

# ------------------ UTILS ------------------

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


# Inicializar historial en session_state
if "historial_individual" not in st.session_state:
    st.session_state.historial_individual = []


def agregar_al_historial(felder, examen, nota):
    st.session_state.historial_individual.append({
        "Felder": felder,
        "Examen": examen,
        "Nota_Estimada": round(float(nota), 3),
    })
    # Mantener últimos 50
    if len(st.session_state.historial_individual) > 50:
        st.session_state.historial_individual = st.session_state.historial_individual[-50:]


# ------------------ CARGA DE ARTEFACTOS ------------------

columnas_one_hot, scaler, model = load_artifacts()

# ------------------ UI PRINCIPAL ------------------

st.title("Predicción de Nota Final - Curso")
st.write(
    "Estima la nota final con el modelo optimizado de Bagging, "
    "de forma individual o cargando un archivo Excel."
)

if not (columnas_one_hot is not None and scaler is not None and model is not None):
    st.warning(
        "Asegúrate de que 'one_hot_columns.joblib', 'min_max_scaler.joblib' y "
        "'bagging_optimizado.joblib' estén en la misma carpeta que este script."
    )
    st.stop()

categorias_felder = [c.replace('Felder_', '') for c in columnas_one_hot if c.startswith('Felder_')]

tab1, tab2, tab3 = st.tabs(["Estudiante individual", "Cargar Excel", "Historial"])

# ------------------ TAB 1: INDIVIDUAL ------------------

with tab1:
    st.markdown('<div class="card fade-in"><h3>Datos del Estudiante</h3></div>', unsafe_allow_html=True)

    felder_sel = st.selectbox("Estilo de Aprendizaje (Felder)", options=categorias_felder, key="felder_ind")
    examen = st.slider("Nota de Examen de Admisión", 0.0, 5.0, 3.8, 0.05, key="examen_ind")

    col_btn, col_empty = st.columns([1, 3])
    with col_btn:
        btn_calc = st.button("Calcular Predicción", key="btn_calc_ind")

    if btn_calc:
        df_in = pd.DataFrame([{COL_FELDER: felder_sel, COL_EXAMEN: examen}])
        X = preparar(df_in, columnas_one_hot, scaler, model)
        pred = model.predict(X)[0]

        st.success(f"### Nota Final Estimada: {pred:.3f}")

        agregar_al_historial(felder_sel, examen, pred)

        with st.expander("Ver variables procesadas enviadas al modelo"):
            st.dataframe(X)

# ------------------ TAB 2: EXCEL MASIVO ------------------

with tab2:
    st.markdown('<div class="card fade-in"><h3>Predicción masiva desde Excel</h3></div>', unsafe_allow_html=True)

    st.write(f"El archivo debe tener las columnas **{COL_FELDER}** y **{COL_EXAMEN}**.")
    st.caption(f"Valores válidos de Felder: {', '.join(categorias_felder)}")

    st.download_button(
        "Descargar plantilla de ejemplo",
        data=plantilla_excel(categorias_felder),
        file_name="plantilla_estudiantes.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        key="btn_plantilla",
    )

    archivo = st.file_uploader("Sube tu archivo Excel", type=["xlsx", "xls"], key="uploader_excel")

    if archivo is not None:
        try:
            xls = pd.ExcelFile(archivo)
            hoja = (
                st.selectbox("Hoja", xls.sheet_names, key="select_hoja")
                if len(xls.sheet_names) > 1
                else xls.sheet_names[0]
            )
            df_excel = pd.read_excel(xls, sheet_name=hoja)
            df_excel.columns = [str(c).strip() for c in df_excel.columns]

            faltantes = [c for c in (COL_FELDER, COL_EXAMEN) if c not in df_excel.columns]
            if faltantes:
                st.error(f"Faltan columnas en el archivo: {', '.join(faltantes)}")
                st.write("Columnas encontradas:", list(df_excel.columns))
            else:
                st.write(f"Filas leídas: {len(df_excel)}")
                st.dataframe(df_excel.head())

                if st.button("Predecir todos", key="btn_predecir_todos"):
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

                    # Estadísticas rápidas
                    notas = resultado["Nota_Final_Estimada"].dropna()
                    if len(notas) > 0:
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.markdown(
                                f'<div class="metric-card"><div class="metric-value">{notas.mean():.3f}</div>'
                                f'<div class="metric-label">Promedio</div></div>',
                                unsafe_allow_html=True,
                            )
                        with col2:
                            st.markdown(
                                f'<div class="metric-card"><div class="metric-value">{notas.median():.3f}</div>'
                                f'<div class="metric-label">Mediana</div></div>',
                                unsafe_allow_html=True,
                            )
                        with col3:
                            st.markdown(
                                f'<div class="metric-card"><div class="metric-value">{notas.min():.3f}</div>'
                                f'<div class="metric-label">Mínima</div></div>',
                                unsafe_allow_html=True,
                            )
                        with col4:
                            st.markdown(
                                f'<div class="metric-card"><div class="metric-value">{notas.max():.3f}</div>'
                                f'<div class="metric-label">Máxima</div></div>',
                                unsafe_allow_html=True,
                            )

                    # Filtros por rango de nota
                    if len(notas) > 0:
                        st.subheader("Filtrar por rango de nota estimada")
                        min_n, max_n = float(notas.min()), float(notas.max())
                        r_min, r_max = st.slider(
                            "Rango de nota",
                            min_value=min_n,
                            max_value=max_n,
                            value=(min_n, max_n),
                            step=0.05,
                            key="slider_rango_notas",
                        )

                        mask_rango = (resultado["Nota_Final_Estimada"] >= r_min) & (
                            resultado["Nota_Final_Estimada"] <= r_max
                        )
                        resultado_filtrado = resultado[mask_rango].copy()

                        st.write(f"Mostrando {len(resultado_filtrado)} de {len(resultado)} filas")

                        st.dataframe(resultado_filtrado)

                        # Descargas
                        st.subheader("Descargar resultados")
                        solo_validos = resultado[resultado["Observación"] == ""].copy()

                        buf_todos = BytesIO()
                        with pd.ExcelWriter(buf_todos, engine="openpyxl") as w:
                            resultado.to_excel(w, index=False, sheet_name="Predicciones")

                        buf_validos = BytesIO()
                        with pd.ExcelWriter(buf_validos, engine="openpyxl") as w:
                            solo_validos.to_excel(w, index=False, sheet_name="Predicciones")

                        c1, c2 = st.columns(2)
                        with c1:
                            st.download_button(
                                "Descargar todos (con inválidos)",
                                data=buf_todos.getvalue(),
                                file_name="predicciones_notas_todos.xlsx",
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                key="btn_download_todos",
                            )
                        with c2:
                            st.download_button(
                                "Descargar solo válidos",
                                data=buf_validos.getvalue(),
                                file_name="predicciones_notas_validos.xlsx",
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                key="btn_download_validos",
                            )

        except Exception as e:
            st.error(f"No se pudo procesar el archivo: {e}")

# ------------------ TAB 3: HISTORIAL ------------------

with tab3:
    st.markdown('<div class="card fade-in"><h3>Historial de predicciones individuales</h3></div>', unsafe_allow_html=True)
    st.write(
        "Aquí puedes ver las últimas predicciones realizadas en la pestaña "
        "“Estudiante individual” durante esta sesión."
    )

    if len(st.session_state.historial_individual) == 0:
        st.info("Aún no hay predicciones en el historial.")
    else:
        df_hist = pd.DataFrame(st.session_state.historial_individual)
        st.dataframe(df_hist)

        buf_hist = BytesIO()
        with pd.ExcelWriter(buf_hist, engine="openpyxl") as w:
            df_hist.to_excel(w, index=False, sheet_name="Historial")
        st.download_button(
            "Descargar historial en Excel",
            data=buf_hist.getvalue(),
            file_name="historial_predicciones.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            key="btn_download_hist",
        )
