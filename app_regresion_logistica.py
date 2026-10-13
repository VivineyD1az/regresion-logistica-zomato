import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, roc_auc_score
)


# ============================================================
# CONFIGURACIÓN
# ============================================================

st.set_page_config(
    page_title="Regresión Logística 🎀",
    page_icon="🎀",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# PALETA DE COLORES
# ============================================================

PINK_MAIN = "#EFA2BE"
PINK_DARK = "#C85D86"
PINK_MEDIUM = "#DD7FA3"
PINK_DEEP = "#B24571"
PINK_SOFT = "#FFEAF2"
PINK_LIGHT = "#FFF7FA"
PINK_BORDER = "#F2C4D5"
TEXT_COLOR = "#5A3D48"
TEXT_LIGHT = "#80636E"

mpl.rcParams.update({
    "text.color": TEXT_COLOR,
    "axes.labelcolor": TEXT_COLOR,
    "axes.edgecolor": PINK_BORDER,
    "axes.titlecolor": PINK_DARK,
    "xtick.color": TEXT_LIGHT,
    "ytick.color": TEXT_LIGHT,
    "legend.edgecolor": PINK_BORDER,
    "font.family": "sans-serif",
})


# ============================================================
# ESTILOS BABY PINK
# ============================================================

st.markdown("""
<style>

:root {
    --pink-main: #EFA2BE;
    --pink-dark: #C85D86;
    --pink-medium: #DD7FA3;
    --pink-soft: #FFEAF2;
    --pink-light: #FFF7FA;
    --pink-very-light: #FFFBFD;
    --pink-border: #F2C4D5;
    --pink-hover: #F8D6E2;
    --text: #5A3D48;
    --text-light: #80636E;
}

.stApp { background: #FFF7FA !important; color: var(--text) !important; }
.main { background: #FFF7FA !important; }
.block-container {
    padding-top: 2rem !important;
    padding-bottom: 4rem !important;
    max-width: 1250px !important;
}

header[data-testid="stHeader"] { background: #FFEAF2 !important; }
header[data-testid="stHeader"] > div { background: #FFEAF2 !important; }
header[data-testid="stHeader"] button { color: #C85D86 !important; }
#MainMenu, footer { visibility: hidden; }

section[data-testid="stSidebar"] {
    background: #FFEAF2 !important;
    border-right: 2px solid #F2C4D5 !important;
}
section[data-testid="stSidebar"] > div { background: #FFEAF2 !important; }
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] span,
section[data-testid="stSidebar"] label { color: #5A3D48 !important; }

h1, h2, h3, h4, h5, h6 { color: #C85D86 !important; }
h1 { font-weight: 800 !important; }
h2 { margin-top: 35px !important; }
p, li, span, div, label { color: #5A3D48; }
.stMarkdown, .stText, .stCaption, .stCaption p { color: #5A3D48 !important; }
label, .stWidgetLabel, div[data-testid="stWidgetLabel"] p {
    color: #6B4A56 !important;
    font-weight: 600 !important;
}

.main-title {
    background: linear-gradient(135deg, #FFEAF2, #FFF7FA, #FFE1EC);
    border: 2px solid #F2C4D5;
    border-radius: 25px;
    padding: 35px;
    margin-bottom: 30px;
    text-align: center;
    box-shadow: 0 6px 20px rgba(200, 93, 134, 0.10);
}
.main-title h1 { font-size: 42px !important; margin-bottom: 10px !important; color: #C85D86 !important; }
.main-title p { font-size: 18px; color: #80636E !important; }

.info-box {
    background: #FFEAF2;
    border-left: 6px solid #EFA2BE;
    border-radius: 15px;
    padding: 20px 25px;
    margin: 20px 0;
}
.info-box h3 { margin-top: 0; }
.soft-box {
    background: #FFF1F6;
    border: 1px solid #F2C4D5;
    border-radius: 18px;
    padding: 20px;
    margin: 15px 0;
}
.card {
    background: white;
    border: 1px solid #F2C4D5;
    border-radius: 18px;
    padding: 22px;
    margin: 10px 0;
    box-shadow: 0 4px 15px rgba(200, 93, 134, 0.07);
}
.card h3 { margin-top: 0; }

div[data-testid="stMetric"] {
    background: #FFEAF2 !important;
    border: 1px solid #F2C4D5 !important;
    border-radius: 18px !important;
    padding: 20px !important;
    box-shadow: 0 3px 10px rgba(200, 93, 134, 0.06);
}
div[data-testid="stMetricLabel"] { color: #80636E !important; }
div[data-testid="stMetricLabel"] p { color: #80636E !important; }
div[data-testid="stMetricValue"] { color: #C85D86 !important; }

.stButton > button, .stDownloadButton > button {
    background: #EFA2BE !important;
    color: white !important;
    border: none !important;
    border-radius: 15px !important;
    padding: 10px 25px !important;
    font-weight: 700 !important;
    transition: 0.2s;
}
.stButton > button:hover, .stDownloadButton > button:hover {
    background: #C85D86 !important;
    color: white !important;
}
.stButton > button p, .stDownloadButton > button p { color: white !important; }

/* SELECTBOX / MULTISELECT (caja cerrada) */
div[data-testid="stSelectbox"] div[data-baseweb="select"],
div[data-testid="stSelectbox"] div[data-baseweb="select"] > div,
div[data-testid="stSelectbox"] div[data-baseweb="select"] div,
div[data-testid="stSelectbox"] div[data-baseweb="base-input"],
div[data-testid="stSelectbox"] div[data-baseweb="base-input"] > div,
div[data-testid="stMultiSelect"] div[data-baseweb="select"],
div[data-testid="stMultiSelect"] div[data-baseweb="select"] > div,
div[data-testid="stMultiSelect"] div[data-baseweb="base-input"],
div[data-testid="stMultiSelect"] div[data-baseweb="base-input"] > div {
    background-color: #FFF7FA !important;
    border-color: #F2C4D5 !important;
    color: #5A3D48 !important;
    box-shadow: none !important;
}
div[data-testid="stSelectbox"] div[data-baseweb="select"],
div[data-testid="stMultiSelect"] div[data-baseweb="select"] {
    border: 2px solid #F2C4D5 !important;
    border-radius: 12px !important;
}
div[data-testid="stSelectbox"] *,
div[data-testid="stMultiSelect"] * { color: #5A3D48 !important; }
div[data-testid="stSelectbox"] svg,
div[data-testid="stMultiSelect"] svg { fill: #C85D86 !important; }
div[data-testid="stSelectbox"] input,
div[data-testid="stMultiSelect"] input {
    color: #5A3D48 !important;
    caret-color: #C85D86 !important;
    background-color: transparent !important;
}
/* Etiquetas (chips) del multiselect */
span[data-baseweb="tag"] {
    background-color: #EFA2BE !important;
    border-radius: 10px !important;
}
span[data-baseweb="tag"] * { color: white !important; }

/* Menú desplegable flotante */
div[data-baseweb="popover"],
div[data-baseweb="popover"] div,
ul[data-testid="stSelectboxVirtualDropdown"] { background-color: #FFFBFD !important; }
div[data-baseweb="popover"] ul[role="listbox"] {
    background: #FFFBFD !important;
    border: 1px solid #F2C4D5 !important;
    border-radius: 12px !important;
}
div[data-baseweb="popover"] li[role="option"],
div[data-baseweb="popover"] div[role="option"] {
    background-color: #FFFBFD !important;
    color: #5A3D48 !important;
}
div[data-baseweb="popover"] li[role="option"]:hover,
div[data-baseweb="popover"] div[role="option"]:hover,
div[data-baseweb="popover"] li[aria-selected="true"],
div[data-baseweb="popover"] div[aria-selected="true"] {
    background-color: #FFEAF2 !important;
    color: #C85D86 !important;
}
div[data-baseweb="popover"] li[role="option"] *,
div[data-baseweb="popover"] div[role="option"] * { color: inherit !important; }

div[data-testid="stNumberInput"] input,
div[data-testid="stTextInput"] input {
    background: #FFF7FA !important;
    color: #5A3D48 !important;
    border: 2px solid #F2C4D5 !important;
    border-radius: 12px !important;
}
div[data-testid="stNumberInput"] button {
    background: #FFEAF2 !important;
    border: 1px solid #F2C4D5 !important;
    color: #C85D86 !important;
}
div[data-testid="stNumberInput"] svg { fill: #C85D86 !important; }

div[data-testid="stSlider"] div[role="slider"] {
    background-color: #C85D86 !important;
    border-color: #C85D86 !important;
}
div[data-testid="stSlider"] .st-emotion-cache-1y4p8pa,
div[data-testid="stTickBar"] { color: #80636E !important; }

section[data-testid="stFileUploaderDropzone"] {
    background: #FFEAF2 !important;
    border: 2px dashed #EFA2BE !important;
    border-radius: 18px !important;
}
section[data-testid="stFileUploaderDropzone"] * { color: #5A3D48 !important; }
section[data-testid="stFileUploaderDropzone"] button {
    background: #EFA2BE !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
}
section[data-testid="stFileUploaderDropzone"] button:hover { background: #C85D86 !important; }
section[data-testid="stFileUploaderDropzone"] button p { color: white !important; }
svg { fill: #C85D86; }

div[data-testid="stExpander"] {
    background: #FFF7FA !important;
    border: 1px solid #F2C4D5 !important;
    border-radius: 15px !important;
}
div[data-testid="stExpander"] summary { color: #C85D86 !important; }
div[data-testid="stExpander"] summary span,
div[data-testid="stExpander"] summary p { color: #C85D86 !important; }

div[data-testid="stAlert"] {
    background: #FFEAF2 !important;
    border: 1px solid #F2C4D5 !important;
    color: #5A3D48 !important;
    border-radius: 14px !important;
}
div[data-testid="stAlert"] p,
div[data-testid="stAlert"] span { color: #5A3D48 !important; }
div[data-testid="stAlert"] svg { fill: #C85D86 !important; }

.pink-table-container {
    width: 100%;
    overflow-x: auto;
    border-radius: 15px;
    margin: 15px 0 25px 0;
}
.pink-table {
    width: 100%;
    border-collapse: separate;
    border-spacing: 0;
    overflow: hidden;
    background: #FFFFFF;
    border: 1px solid #F2C4D5;
    border-radius: 15px;
    color: #5A3D48;
    font-size: 14px;
}
.pink-table thead th {
    background: #EFA2BE !important;
    color: white !important;
    padding: 13px 15px;
    text-align: center;
    font-weight: 700;
    border-bottom: 2px solid #D77D9D;
}
.pink-table tbody td {
    background: #FFF7FA !important;
    color: #5A3D48 !important;
    padding: 11px 15px;
    text-align: center;
    border-bottom: 1px solid #F5D9E3;
}
.pink-table tbody tr:nth-child(even) td { background: #FFEAF2 !important; }
.pink-table tbody tr:hover td { background: #F8D6E2 !important; }
.pink-table tbody tr:last-child td { border-bottom: none; }

.equation-box {
    background: #FFEAF2;
    border: 2px solid #EFA2BE;
    border-radius: 18px;
    padding: 25px;
    text-align: center;
    margin: 20px 0;
}
.equation { font-size: 25px; font-weight: bold; color: #C85D86; }

hr {
    border: none !important;
    height: 2px !important;
    background: #F2C4D5 !important;
    margin: 35px 0 !important;
}

.footer {
    margin-top: 50px;
    padding: 25px;
    text-align: center;
    background: #FFEAF2;
    border-top: 2px solid #F2C4D5;
    border-radius: 20px;
}
.footer p { margin: 3px; color: #80636E !important; }

::-webkit-scrollbar { width: 10px; }
::-webkit-scrollbar-track { background: #FFF7FA; }
::-webkit-scrollbar-thumb { background: #EFA2BE; border-radius: 10px; }

</style>
""", unsafe_allow_html=True)


# ============================================================
# FUNCIONES
# ============================================================

def pink_table(df, max_rows=None, index=True):
    if max_rows:
        df = df.head(max_rows)
    df = df.copy()
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].round(4)
    html = df.to_html(index=index, classes="pink-table", border=0)
    st.markdown(
        f'<div class="pink-table-container">{html}</div>',
        unsafe_allow_html=True
    )


def section_title(number, title, emoji):
    st.markdown(f"<h2>{emoji} {number}. {title}</h2>", unsafe_allow_html=True)


def style_axes(ax):
    for spine in ax.spines.values():
        spine.set_color(PINK_BORDER)
    ax.tick_params(colors=TEXT_LIGHT)
    ax.xaxis.label.set_color(TEXT_COLOR)
    ax.yaxis.label.set_color(TEXT_COLOR)
    legend = ax.get_legend()
    if legend:
        legend.get_frame().set_facecolor(PINK_LIGHT)
        legend.get_frame().set_edgecolor(PINK_BORDER)
        for text in legend.get_texts():
            text.set_color(TEXT_COLOR)


@st.cache_data
def load_csv(file_or_path):
    """Lee el CSV probando varias codificaciones (el de Zomato trae
    símbolos de moneda que a veces fallan con utf-8)."""
    for enc in ("utf-8-sig", "latin-1"):
        try:
            if hasattr(file_or_path, "seek"):
                file_or_path.seek(0)
            return pd.read_csv(file_or_path, encoding=enc)
        except UnicodeDecodeError:
            continue
    raise ValueError("No se pudo decodificar el archivo.")


# ============================================================
# TÍTULO
# ============================================================

st.markdown("""
<div class="main-title">

<h1>🎀 Análisis de Regresión Logística</h1>

<p>
Una aplicación interactiva para explorar datos de restaurantes,
construir un modelo de clasificación y evaluar qué tan bien
predice una categoría (Sí / No).
</p>

</div>
""", unsafe_allow_html=True)


# ============================================================
# INTRODUCCIÓN
# ============================================================

st.markdown("""
<div class="info-box">

<h3>🌸 ¿Qué encontrarás en esta aplicación?</h3>

<p>
Esta aplicación utiliza <b>regresión logística</b> sobre el dataset
de restaurantes de Zomato. A diferencia de la regresión lineal,
aquí la variable que queremos predecir es <b>categórica</b>
(por ejemplo: ¿tiene entrega a domicilio? Sí / No).
</p>

<p>
Podrás elegir la variable objetivo y las variables predictoras,
entrenar el modelo, revisar sus métricas, la <b>matriz de confusión</b>
y la <b>matriz de densidad</b> de las probabilidades predichas.
</p>

</div>
""", unsafe_allow_html=True)


# ============================================================
# EXPLICACIÓN
# ============================================================

section_title("1", "¿Qué es la regresión logística?", "📚")

st.markdown("""
<div class="card">

<p>
La <b>regresión logística</b> es una técnica de clasificación que estima
la <b>probabilidad</b> de que una observación pertenezca a la clase
positiva (1) en lugar de la negativa (0).
</p>

<p>
Primero calcula una combinación lineal de las variables y luego la pasa
por la <b>función sigmoide</b>, que convierte cualquier número en una
probabilidad entre 0 y 1:
</p>

</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="equation-box">

<div class="equation">
P(Y=1) = 1 / (1 + e<sup>−(β₀ + β₁X₁ + … + βₖXₖ)</sup>)
</div>

<p>
Si la probabilidad es mayor o igual al <b>umbral</b> (normalmente 0.5),
el modelo clasifica la observación como clase 1; de lo contrario, como clase 0.
</p>

</div>
""", unsafe_allow_html=True)


# ============================================================
# DATOS
# ============================================================

section_title("2", "Carga y exploración de los datos", "📂")

st.markdown("""
<div class="soft-box">

<b>💗 ¿Qué hacer aquí?</b><br>

Puedes usar el archivo <code>Zomato_Restaurant_Dataset.csv</code> del
proyecto o cargar otro CSV. La aplicación detecta automáticamente las
variables numéricas y las columnas con dos categorías (Sí / No).

</div>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "🎀 Selecciona un archivo CSV",
    type=["csv"]
)

df = None

if uploaded_file is not None:
    try:
        df = load_csv(uploaded_file)
        st.success("✨ Archivo cargado correctamente.")
    except Exception as e:
        st.error(f"No fue posible leer el archivo: {e}")
else:
    try:
        df = load_csv("Zomato_Restaurant_Dataset.csv")
    except Exception:
        st.error(
            "No se encontró 'Zomato_Restaurant_Dataset.csv' junto a la "
            "aplicación. Colócalo en la misma carpeta o súbelo arriba."
        )


# ============================================================
# TODO LO DEMÁS SOLO SI HAY DATOS
# ============================================================

if df is not None:

    # ----- Resumen -----
    numeric_columns_all = df.select_dtypes(include=np.number).columns.tolist()

    st.markdown("""
    <div class="card">
    <h3>📋 Resumen del conjunto de datos</h3>
    <p>
    Cada fila es un restaurante. Usaremos sus características
    para clasificar una categoría de interés.
    </p>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("📊 Registros", f"{len(df):,}")
    with c2:
        st.metric("📝 Variables", len(df.columns))
    with c3:
        st.metric("🔢 Numéricas", len(numeric_columns_all))
    with c4:
        st.metric("⚠️ Valores faltantes", int(df.isna().sum().sum()))

    st.markdown("### 🌷 Vista previa de los datos")
    pink_table(df, 10)

    with st.expander("🌸 Ver estadísticas descriptivas"):
        pink_table(df.describe().T)


    # ========================================================
    # 3. VARIABLE OBJETIVO
    # ========================================================

    st.markdown("<hr>", unsafe_allow_html=True)
    section_title("3", "Variable objetivo (Y)", "🎯")

    st.markdown("""
    <div class="info-box">

    <h3>💗 ¿Qué queremos predecir?</h3>

    <p>
    La regresión logística necesita una variable <b>binaria</b>
    (dos posibles resultados). Puedes elegir una columna Sí / No del
    dataset, o crear la variable <b>«Rating alto»</b>, que vale 1 si la
    calificación del restaurante es mayor o igual a un umbral.
    </p>

    </div>
    """, unsafe_allow_html=True)

    RATING_TARGET = "Rating alto"

    # Columnas con exactamente 2 valores distintos (Yes/No, etc.)
    binary_columns = [
        c for c in df.columns
        if df[c].nunique(dropna=True) == 2
    ]

    target_options = binary_columns.copy()
    if "Aggregate rating" in df.columns:
        target_options.append(RATING_TARGET)

    if not target_options:
        st.error("El dataset no tiene columnas binarias para usar como objetivo.")
        st.stop()

    default_target = (
        target_options.index(RATING_TARGET)
        if RATING_TARGET in target_options
        else 0
    )

    target_name = st.selectbox(
        "🎀 Variable objetivo (Y)",
        target_options,
        index=default_target
    )

    work_df = df.copy()
    excluded_from_X = set()
    target_labels = {0: "0", 1: "1"}

    if target_name == RATING_TARGET:

        # Solo restaurantes calificados (rating 0 = "Not rated")
        rated_only = st.checkbox(
            "Excluir restaurantes sin calificación (rating = 0)",
            value=True
        )
        threshold_rating = st.slider(
            "Umbral para considerar un rating como «alto»",
            min_value=2.5, max_value=4.5, value=4.0, step=0.1
        )

        if rated_only:
            work_df = work_df[work_df["Aggregate rating"] > 0]

        work_df[RATING_TARGET] = (
            work_df["Aggregate rating"] >= threshold_rating
        ).astype(int)

        # Evitar fuga de información: el rating define el objetivo
        excluded_from_X.update(["Aggregate rating"])
        target_labels = {0: "Rating bajo", 1: "Rating alto"}

    else:
        values = sorted(work_df[target_name].dropna().unique().tolist(), key=str)

        positive_label = st.selectbox(
            "🌸 ¿Cuál valor se considera la clase positiva (1)?",
            values,
            index=(values.index("Yes") if "Yes" in values else len(values) - 1)
        )
        negative_label = [v for v in values if v != positive_label][0]

        work_df = work_df[work_df[target_name].notna()]
        work_df[target_name] = (
            work_df[target_name] == positive_label
        ).astype(int)

        target_labels = {0: str(negative_label), 1: str(positive_label)}

        if target_name == "Aggregate rating":
            excluded_from_X.add("Aggregate rating")

    # Distribución de clases
    class_counts = work_df[target_name].value_counts().sort_index()

    if len(class_counts) < 2:
        st.error("Con esta configuración solo queda una clase. Cambia las opciones.")
        st.stop()

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("### 💕 Distribución de las clases")
        dist_df = pd.DataFrame({
            "Clase": [f"{k} = {target_labels[k]}" for k in class_counts.index],
            "Registros": class_counts.values,
            "Porcentaje (%)": (class_counts.values / class_counts.sum() * 100).round(2)
        })
        pink_table(dist_df, index=False)

        minority = class_counts.min() / class_counts.sum()
        if minority < 0.2:
            st.info(
                "💗 Las clases están desbalanceadas. Puede ser útil activar "
                "«Balancear clases» más abajo."
            )

    with col2:
        fig, ax = plt.subplots(figsize=(5, 3.4))
        fig.patch.set_facecolor(PINK_LIGHT)
        ax.set_facecolor(PINK_LIGHT)
        ax.bar(
            [target_labels[k] for k in class_counts.index],
            class_counts.values,
            color=[PINK_MAIN, PINK_DEEP][:len(class_counts)],
            edgecolor=PINK_DARK
        )
        ax.set_ylabel("Registros")
        ax.set_title("Clases de la variable objetivo", fontweight="bold")
        ax.grid(axis="y", alpha=0.2, color=PINK_BORDER)
        style_axes(ax)
        st.pyplot(fig)


    # ========================================================
    # 4. CORRELACIÓN
    # ========================================================

    st.markdown("<hr>", unsafe_allow_html=True)
    section_title("4", "Análisis de correlación", "📊")

    st.markdown("""
    <div class="card">
    <p>
    La <b>correlación</b> mide qué tan relacionadas están dos variables
    numéricas (de −1 a 1). Aquí incluimos también la variable objetivo
    (0/1) para ver qué variables se asocian más con ella.
    </p>
    </div>
    """, unsafe_allow_html=True)

    corr_cols = [
        c for c in work_df.select_dtypes(include=np.number).columns
        if c not in ("Restaurant ID", "Switch to order menu")
    ]
    if target_name not in corr_cols:
        corr_cols.append(target_name)

    correlation = work_df[corr_cols].corr()

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("### 💕 Correlación con la variable objetivo")
        corr_target = (
            correlation[target_name]
            .drop(target_name)
            .sort_values(ascending=False)
            .to_frame("Correlación")
        )
        pink_table(corr_target)

    with col2:
        st.markdown("### 🎀 Mapa de calor")
        fig, ax = plt.subplots(figsize=(7, 5.5))
        fig.patch.set_facecolor(PINK_LIGHT)
        ax.set_facecolor(PINK_LIGHT)
        sns.heatmap(
            correlation, annot=True, cmap="RdPu", fmt=".2f",
            linewidths=1, linecolor="#FFFFFF", ax=ax,
            annot_kws={"color": TEXT_COLOR, "size": 8},
            cbar_kws={"label": ""}
        )
        ax.set_title("Matriz de correlación", color=PINK_DARK,
                     fontsize=14, fontweight="bold")
        style_axes(ax)
        cbar = ax.collections[0].colorbar
        cbar.ax.yaxis.set_tick_params(color=TEXT_LIGHT)
        plt.setp(plt.getp(cbar.ax.axes, "yticklabels"), color=TEXT_LIGHT)
        st.pyplot(fig)


    # ========================================================
    # 5. VARIABLES PREDICTORAS
    # ========================================================

    st.markdown("<hr>", unsafe_allow_html=True)
    section_title("5", "Selección de variables predictoras", "🌷")

    st.markdown("""
    <div class="info-box">

    <h3>💗 ¿Con qué variables predecimos?</h3>

    <p>
    Elige una o varias variables numéricas como predictoras (X).
    El modelo las estandariza automáticamente para que tengan
    la misma escala.
    </p>

    </div>
    """, unsafe_allow_html=True)

    # Columnas que no tienen sentido como predictoras
    id_like = {"Restaurant ID", "Switch to order menu", target_name}

    candidate_features = [
        c for c in work_df.select_dtypes(include=np.number).columns
        if c not in id_like and c not in excluded_from_X
    ]

    default_features = [
        c for c in ["Votes", "Price range", "Aggregate rating"]
        if c in candidate_features
    ] or candidate_features[:2]

    if excluded_from_X:
        st.caption(
            "ℹ️ «Aggregate rating» no se ofrece como predictora porque "
            "define la variable objetivo (sería «hacer trampa»)."
        )

    features = st.multiselect(
        "🌸 Variables independientes (X)",
        candidate_features,
        default=default_features
    )

    if "Average Cost for two" in features:
        st.warning(
            "💗 «Average Cost for two» está en monedas distintas según el "
            "país, así que sus valores no son comparables entre sí. "
            "Considera usar «Price range» en su lugar."
        )

    if len(features) == 0:
        st.warning("💗 Selecciona al menos una variable predictora.")
        st.stop()


    # ========================================================
    # 6. MODELO
    # ========================================================

    st.markdown("<hr>", unsafe_allow_html=True)
    section_title("6", "Construcción del modelo", "📈")

    st.markdown("""
    <div class="card">
    <p>
    Los datos se dividen en un conjunto de <b>entrenamiento</b> (con el que
    el modelo aprende) y uno de <b>prueba</b> (datos que nunca vio, para
    evaluar si generaliza bien).
    </p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        test_size = st.slider(
            "🎀 Porcentaje para prueba (%)", 10, 40, 25, step=5
        ) / 100

    with col2:
        threshold = st.slider(
            "🌸 Umbral de clasificación", 0.05, 0.95, 0.50, step=0.05
        )

    with col3:
        balance = st.checkbox("💗 Balancear clases", value=False)

    model_df = work_df[features + [target_name]].dropna()

    X = model_df[features]
    y = model_df[target_name].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=42,
        stratify=y
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced" if balance else None
    )
    model.fit(X_train_s, y_train)

    proba_test = model.predict_proba(X_test_s)[:, 1]
    pred_test = (proba_test >= threshold).astype(int)

    accuracy = accuracy_score(y_test, pred_test)
    precision = precision_score(y_test, pred_test, zero_division=0)
    recall = recall_score(y_test, pred_test, zero_division=0)
    f1 = f1_score(y_test, pred_test, zero_division=0)
    auc = roc_auc_score(y_test, proba_test)

    st.markdown("### 💕 Resultados del modelo (conjunto de prueba)")

    m1, m2, m3, m4, m5 = st.columns(5)
    with m1:
        st.metric("🎀 Exactitud", f"{accuracy:.3f}")
    with m2:
        st.metric("🌸 Precisión", f"{precision:.3f}")
    with m3:
        st.metric("💗 Sensibilidad", f"{recall:.3f}")
    with m4:
        st.metric("🌷 F1", f"{f1:.3f}")
    with m5:
        st.metric("✨ AUC", f"{auc:.3f}")

    st.markdown(f"""
    <div class="soft-box">
    <b>💗 ¿Cómo leer estas métricas?</b><br>
    <b>Exactitud:</b> porcentaje de aciertos totales.<br>
    <b>Precisión:</b> de los que el modelo dijo «{target_labels[1]}», cuántos realmente lo eran.<br>
    <b>Sensibilidad (recall):</b> de los que realmente eran «{target_labels[1]}», cuántos detectó.<br>
    <b>F1:</b> balance entre precisión y sensibilidad.<br>
    <b>AUC:</b> capacidad de separar las dos clases (0.5 = azar, 1 = perfecto).
    </div>
    """, unsafe_allow_html=True)

    # Ecuación
    st.markdown("### 🧮 Ecuación del modelo (variables estandarizadas)")

    terms = " ".join(
        f"{'+' if c >= 0 else '-'} {abs(c):.3f}·z({f})"
        for f, c in zip(features, model.coef_[0])
    )
    equation = f"logit(P) = {model.intercept_[0]:.3f} {terms}"

    st.markdown(
        f"""
        <div class="equation-box">
        <div class="equation" style="font-size:18px;">{equation}</div>
        <p>
        Donde <b>logit(P) = ln(P / (1 − P))</b> y z(X) es la variable estandarizada.
        </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Coeficientes y odds ratio
    st.markdown("### 🌷 Coeficientes y razón de momios (odds ratio)")

    coef_df = pd.DataFrame({
        "Variable": features,
        "Coeficiente": model.coef_[0],
        "Odds ratio": np.exp(model.coef_[0])
    }).sort_values("Coeficiente", ascending=False)

    pink_table(coef_df, index=False)


    # ========================================================
    # 7. INTERPRETACIÓN
    # ========================================================

    st.markdown("<hr>", unsafe_allow_html=True)
    section_title("7", "Interpretación de los resultados", "🔎")

    strongest = coef_df.iloc[coef_df["Coeficiente"].abs().argmax()]
    direction = "aumenta" if strongest["Coeficiente"] > 0 else "disminuye"

    st.markdown(
        f"""
        <div class="card">

        <h3>🌸 Interpretación de los coeficientes</h3>

        <p>
        Un coeficiente positivo indica que al aumentar la variable
        sube la probabilidad de la clase <b>«{target_labels[1]}»</b>;
        uno negativo indica lo contrario.
        </p>

        <p>
        La variable con mayor efecto es <b>{strongest['Variable']}</b>:
        por cada aumento de una desviación estándar, los momios
        (odds) de «{target_labels[1]}» se multiplican por
        <b>{strongest['Odds ratio']:.3f}</b>, es decir, la probabilidad
        <b>{direction}</b>.
        </p>

        <h3>🎀 Interpretación del AUC</h3>

        <p>
        El AUC es <b>{auc:.3f}</b>. Si tomamos al azar un restaurante de
        cada clase, el modelo asigna una probabilidad mayor al de la
        clase «{target_labels[1]}» en aproximadamente
        <b>{auc*100:.1f}%</b> de los casos.
        </p>

        </div>
        """,
        unsafe_allow_html=True
    )


    # ========================================================
    # 8. MATRIZ DE CONFUSIÓN
    # ========================================================

    st.markdown("<hr>", unsafe_allow_html=True)
    section_title("8", "Matriz de confusión", "🧩")

    st.markdown("""
    <div class="soft-box">

    <b>💗 ¿Qué muestra esta matriz?</b><br>

    Compara lo que el modelo predijo con lo que realmente ocurrió.
    La diagonal son los <b>aciertos</b>; fuera de la diagonal están
    los <b>errores</b>. Se muestran los conteos y el porcentaje
    por fila (qué fracción de cada clase real se clasificó bien).

    </div>
    """, unsafe_allow_html=True)

    cm = confusion_matrix(y_test, pred_test, labels=[0, 1])
    cm_pct = cm / cm.sum(axis=1, keepdims=True) * 100

    annot = np.array([
        [f"{cm[i, j]}\n({cm_pct[i, j]:.1f}%)" for j in range(2)]
        for i in range(2)
    ])

    label_names = [target_labels[0], target_labels[1]]

    col1, col2 = st.columns([1.2, 1])

    with col1:
        fig, ax = plt.subplots(figsize=(6.5, 5))
        fig.patch.set_facecolor(PINK_LIGHT)
        ax.set_facecolor(PINK_LIGHT)

        sns.heatmap(
            cm, annot=annot, fmt="", cmap="RdPu",
            xticklabels=label_names, yticklabels=label_names,
            linewidths=2, linecolor="#FFFFFF", ax=ax,
            annot_kws={"color": TEXT_COLOR, "size": 13, "weight": "bold"},
            cbar_kws={"label": ""}
        )
        ax.set_xlabel("Predicción del modelo")
        ax.set_ylabel("Valor real")
        ax.set_title("Matriz de confusión", color=PINK_DARK,
                     fontsize=14, fontweight="bold")
        style_axes(ax)
        cbar = ax.collections[0].colorbar
        cbar.ax.yaxis.set_tick_params(color=TEXT_LIGHT)
        plt.setp(plt.getp(cbar.ax.axes, "yticklabels"), color=TEXT_LIGHT)
        st.pyplot(fig)

    with col2:
        tn, fp, fn, tp = cm.ravel()
        st.markdown(
            f"""
            <div class="card">
            <h3>🌸 Lectura de la matriz</h3>
            <p><b>Verdaderos negativos:</b> {tn}<br>
            <b>Falsos positivos:</b> {fp}<br>
            <b>Falsos negativos:</b> {fn}<br>
            <b>Verdaderos positivos:</b> {tp}</p>
            <p>
            Los <b>falsos positivos</b> son restaurantes «{target_labels[0]}»
            que el modelo clasificó como «{target_labels[1]}»; los
            <b>falsos negativos</b> son lo contrario.
            </p>
            </div>
            """,
            unsafe_allow_html=True
        )


    # ========================================================
    # 9. MATRIZ DE DENSIDAD
    # ========================================================

    st.markdown("<hr>", unsafe_allow_html=True)
    section_title("9", "Matriz de densidad", "🌈")

    st.markdown("""
    <div class="soft-box">

    <b>💗 ¿Qué muestra esta sección?</b><br>

    La <b>densidad</b> indica dónde se concentran las observaciones.
    A la izquierda ves cómo se distribuyen las <b>probabilidades predichas</b>
    para cada clase real: si las dos curvas están bien separadas, el modelo
    distingue bien las clases; si se traslapan, comete más errores.
    A la derecha ves un mapa de densidad entre dos variables.

    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    # ---- Densidad de probabilidades ----
    with col1:
        fig, ax = plt.subplots(figsize=(7, 5))
        fig.patch.set_facecolor(PINK_LIGHT)
        ax.set_facecolor(PINK_LIGHT)

        dens_df = pd.DataFrame({
            "prob": proba_test,
            "clase": [target_labels[v] for v in y_test.values]
        })

        palette = {
            target_labels[0]: PINK_MAIN,
            target_labels[1]: PINK_DEEP
        }

        try:
            sns.kdeplot(
                data=dens_df, x="prob", hue="clase",
                fill=True, alpha=0.45, common_norm=False,
                palette=palette, linewidth=2, clip=(0, 1), ax=ax
            )
        except Exception:
            for cls, color in palette.items():
                ax.hist(
                    dens_df.loc[dens_df["clase"] == cls, "prob"],
                    bins=25, density=True, alpha=0.5,
                    color=color, label=cls
                )
            ax.legend()

        ax.axvline(threshold, color=PINK_DARK, linestyle="--", linewidth=2)
        ax.text(
            threshold, ax.get_ylim()[1] * 0.95,
            f" umbral = {threshold:.2f}",
            color=PINK_DARK, va="top", fontsize=10
        )
        ax.set_xlim(0, 1)
        ax.set_xlabel(f"Probabilidad predicha de «{target_labels[1]}»")
        ax.set_ylabel("Densidad")
        ax.set_title("Densidad de probabilidades por clase real",
                     fontsize=13, fontweight="bold")
        ax.grid(alpha=0.2, color=PINK_BORDER)
        style_axes(ax)
        st.pyplot(fig)

    # ---- Densidad 2D entre dos variables ----
    with col2:
        if len(features) >= 2:
            dx = st.selectbox("🌸 Variable en el eje X", features, index=0, key="dx")
            dy = st.selectbox(
                "🎀 Variable en el eje Y", features,
                index=1 if len(features) > 1 else 0, key="dy"
            )
        else:
            dx = dy = features[0]
            st.caption("Selecciona 2 o más variables predictoras para ver la densidad conjunta.")

        fig, ax = plt.subplots(figsize=(7, 5))
        fig.patch.set_facecolor(PINK_LIGHT)
        ax.set_facecolor(PINK_LIGHT)

        if dx != dy:
            sample = model_df.sample(min(3000, len(model_df)), random_state=42)

            # Los percentiles 1–99 evitan que unos pocos valores extremos
            # (por ejemplo restaurantes con miles de votos) aplasten el gráfico.
            lo_x, hi_x = sample[dx].quantile([0.01, 0.99])
            lo_y, hi_y = sample[dy].quantile([0.01, 0.99])
            sample = sample[
                sample[dx].between(lo_x, hi_x) & sample[dy].between(lo_y, hi_y)
            ]

            try:
                sns.kdeplot(
                    data=sample, x=dx, y=dy,
                    fill=True, cmap="RdPu", thresh=0.05, levels=10, ax=ax
                )
                ax.set_title(
                    f"Densidad conjunta: {dx} vs {dy}",
                    fontsize=13, fontweight="bold"
                )
            except Exception:
                ax.hexbin(sample[dx], sample[dy], gridsize=25, cmap="RdPu")
                ax.set_title(
                    f"Densidad (hexbin): {dx} vs {dy}",
                    fontsize=13, fontweight="bold"
                )
            ax.set_xlabel(dx)
            ax.set_ylabel(dy)
        else:
            sns.kdeplot(
                model_df[dx], fill=True, color=PINK_MEDIUM, ax=ax
            )
            ax.set_title(f"Densidad de {dx}", fontsize=13, fontweight="bold")

        ax.grid(alpha=0.2, color=PINK_BORDER)
        style_axes(ax)
        st.pyplot(fig)


    # ========================================================
    # 10. CURVA ROC
    # ========================================================

    st.markdown("<hr>", unsafe_allow_html=True)
    section_title("10", "Curva ROC", "📉")

    st.markdown("""
    <div class="soft-box">

    <b>💗 ¿Qué muestra este gráfico?</b><br>

    La curva ROC muestra el compromiso entre detectar más positivos
    reales (eje Y) y cometer más falsas alarmas (eje X) al mover el umbral.
    Mientras más se aleje de la diagonal, mejor es el modelo.

    </div>
    """, unsafe_allow_html=True)

    fpr, tpr, _ = roc_curve(y_test, proba_test)

    fig, ax = plt.subplots(figsize=(7, 5.5))
    fig.patch.set_facecolor(PINK_LIGHT)
    ax.set_facecolor(PINK_LIGHT)

    ax.plot(fpr, tpr, color=PINK_DEEP, linewidth=3, label=f"Modelo (AUC = {auc:.3f})")
    ax.fill_between(fpr, tpr, alpha=0.15, color=PINK_MAIN)
    ax.plot([0, 1], [0, 1], linestyle="--", color=TEXT_LIGHT, label="Azar")
    ax.set_xlabel("Tasa de falsos positivos")
    ax.set_ylabel("Tasa de verdaderos positivos")
    ax.set_title("Curva ROC", fontsize=15, fontweight="bold")
    ax.grid(alpha=0.2, color=PINK_BORDER)
    ax.legend()
    style_axes(ax)
    st.pyplot(fig)


    # ========================================================
    # 11. PREDICCIÓN MANUAL
    # ========================================================

    st.markdown("<hr>", unsafe_allow_html=True)
    section_title("11", "Realizar una predicción", "🧮")

    st.markdown("""
    <div class="info-box">

    <h3>🌷 Prueba el modelo</h3>

    <p>
    Introduce valores para las variables predictoras y observa la
    probabilidad que estima el modelo y la clase que asigna.
    </p>

    </div>
    """, unsafe_allow_html=True)

    user_values = {}
    cols = st.columns(min(len(features), 3))

    for i, f in enumerate(features):
        with cols[i % len(cols)]:
            user_values[f] = st.number_input(
                f"🎀 {f}",
                min_value=float(X[f].min()),
                max_value=float(X[f].max()),
                value=float(X[f].median())
            )

    user_df = pd.DataFrame([user_values])
    user_proba = model.predict_proba(scaler.transform(user_df))[0, 1]
    user_class = int(user_proba >= threshold)

    st.markdown(
        f"""
        <div class="equation-box">

        <p>Probabilidad estimada de «{target_labels[1]}»:</p>

        <div class="equation">{user_proba*100:.1f}%</div>

        <p>Con un umbral de {threshold:.2f}, el modelo clasifica como:</p>

        <div class="equation">{target_labels[user_class]}</div>

        </div>
        """,
        unsafe_allow_html=True
    )


    # ========================================================
    # 12. CONCLUSIÓN
    # ========================================================

    st.markdown("<hr>", unsafe_allow_html=True)
    section_title("12", "Conclusión", "💡")

    st.markdown(
        f"""
        <div class="info-box">

        <h3>🎀 Conclusión del análisis</h3>

        <p>
        Se construyó un modelo de <b>regresión logística</b> para clasificar
        la variable <b>{target_name}</b> usando
        <b>{", ".join(features)}</b> como predictoras.
        </p>

        <p>
        En el conjunto de prueba el modelo obtuvo una exactitud de
        <b>{accuracy:.3f}</b>, una precisión de <b>{precision:.3f}</b>,
        una sensibilidad de <b>{recall:.3f}</b> y un AUC de
        <b>{auc:.3f}</b>.
        </p>

        <p>
        La matriz de confusión y la matriz de densidad permiten ver en
        qué casos el modelo acierta y dónde se confunde, según qué tan
        separadas están las probabilidades de cada clase.
        </p>

        <p>
        Es importante recordar que el modelo describe asociaciones
        estadísticas y no implica causalidad.
        </p>

        </div>
        """,
        unsafe_allow_html=True
    )


    # ========================================================
    # FOOTER
    # ========================================================

    st.markdown("""
    <div class="footer">

    <p>🎀 <b>Proyecto de Regresión Logística</b></p>

    <p>Análisis de datos · Clasificación · Matriz de confusión · Densidad</p>

    <p>💗 Aplicación desarrollada con Python y Streamlit</p>

    <p>Hecho por Mishelle, Viviney y Ana</p>

    </div>
    """, unsafe_allow_html=True)
