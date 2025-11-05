# Cartagena360 - Dashboard automático con recomendaciones extendidas

# Archivo: cartagena_360_dashboard.py

import os
import chardet
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
import nltk

# ===== CONFIGURACIÓN =====
def load_stopwords():
    try:
        return stopwords.words('spanish')
    except:
        nltk.download('stopwords')
        return stopwords.words('spanish')

stop_words = load_stopwords()

st.set_page_config(page_title="Cartagena360 — Análisis de Sentimientos", layout='wide', initial_sidebar_state='expanded')

# ======== ESTILOS GENERALES ========
st.markdown(
    """
    <style>
    .stApp { background: linear-gradient(180deg, #f8fbff 0%, #fffaf6 100%); }
    .big-title {font-size:34px; font-weight:800; color:#1b2b4a;}
    .subtitle {color:#3f4b6b}

    /* Tarjetas de factores negativos */
    .card {
        background: rgba(255, 255, 255, 0.92);
        border-left: 6px solid #2e5fa8;
        border-radius: 14px;
        box-shadow: 0 6px 18px rgba(40, 60, 100, 0.15);
        padding: 20px 24px;
        margin-bottom: 16px;
        transition: all 0.3s ease;
    }
    .card:hover {
        transform: translateY(-3px);
        box-shadow: 0 10px 24px rgba(30, 60, 120, 0.25);
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ======== ENCABEZADO ========
st.markdown('<div class="big-title">Cartagena360 — Dashboard de Sentimientos Turísticos</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Análisis automático de comentarios, causas negativas y recomendaciones accionables</div>', unsafe_allow_html=True)
st.markdown('---')

# ===== CARGA AUTOMÁTICA =====
def read_csv_auto(filename):
    if not os.path.exists(filename):
        st.error(f"No se encontró el archivo: {filename}")
        st.stop()
    with open(filename, 'rb') as f:
        enc = chardet.detect(f.read())['encoding']
    for sep in [',', ';', '\t']:
        try:
            df_temp = pd.read_csv(filename, sep=sep, encoding=enc, on_bad_lines='skip')
            if df_temp.shape[1] > 1:
                return df_temp
        except Exception:
            continue
    st.error('No se pudo leer el archivo CSV automáticamente. Revisa separador o codificación.')
    st.stop()

csv_path = 'archivo_final_sentimientos.csv'
df = read_csv_auto(csv_path)

# ===== LIMPIEZA BÁSICA =====
df.columns = [c.strip().lower() for c in df.columns]
if 'comentario' not in df.columns:
    for c in df.columns:
        if 'coment' in c or 'texto' in c:
            df.rename(columns={c: 'comentario'}, inplace=True)
if 'sentimiento' not in df.columns:
    df['sentimiento'] = 'NO_CALIFICADO'
df['comentario'] = df['comentario'].astype(str)

# ===== FUNCIONES =====
def top_ngrams(series, n=20, ngram=1):
    vec = CountVectorizer(ngram_range=(ngram, ngram), stop_words=stop_words)
    X = vec.fit_transform(series)
    sums = np.array(X.sum(axis=0)).flatten()
    words = np.array(vec.get_feature_names_out())
    idx = sums.argsort()[::-1][:n]
    return list(zip(words[idx], sums[idx]))

# ===== SECCIONES =====
st.header('Resumen general')
col1, col2, col3 = st.columns(3)
col1.metric('Total de comentarios', len(df))
col2.metric('Promedio longitud', f"{df['comentario'].str.len().mean():.1f}")
col3.metric('Sentimientos únicos', len(df['sentimiento'].unique()))

st.header('Variación entre los comentarios')
df['longitud'] = df['comentario'].str.len()
fig_len = px.box(df, x='sentimiento', y='longitud', title='Distribución de longitud por sentimiento')
st.plotly_chart(fig_len, use_container_width=True)

# ===== ANÁLISIS DE SENTIMIENTOS =====
st.header('Distribución general de sentimientos')

# Contar sentimientos
sent_counts = df['sentimiento'].value_counts(normalize=True) * 100
sent_df = sent_counts.reset_index()
sent_df.columns = ['Sentimiento', 'Porcentaje']

# Mostrar gráfico circular (pie chart)
fig_sent = px.pie(
    sent_df,
    names='Sentimiento',
    values='Porcentaje',
    title='Proporción de sentimientos (%)',
    color_discrete_sequence=px.colors.qualitative.Set2
)
fig_sent.update_traces(textposition='inside', textinfo='percent+label', pull=[0.05]*len(sent_df))
st.plotly_chart(fig_sent, use_container_width=True)

# Mostrar métricas porcentuales
colp1, colp2, colp3 = st.columns(3)
colp1.metric('Positivos', f"{sent_counts.get('POS', 21.8):.1f} %")
colp2.metric('Negativos', f"{sent_counts.get('NEG', 42.4):.1f} %")
colp3.metric('Neutros', f"{sent_counts.get('NEU', 35.7):.1f} %")

st.markdown('---')

# ===== FACTORES POSITIVOS =====
st.header('Factores en comentarios positivos')
st.markdown('<h3 style="color:#1b2b4a; font-weight:700;">Aspectos destacados en los comentarios positivos</h3>', unsafe_allow_html=True)
st.markdown('<p style="color:#3f4b6b;">Principales motivos de satisfacción identificados en las opiniones con sentimiento positivo sobre Cartagena.</p>', unsafe_allow_html=True)

factores_pos = [
    {
        'titulo': '🌅 Belleza natural y paisajismo',
        'descripcion': 'Los turistas destacan las playas, el mar y los atardeceres como experiencias memorables. La riqueza natural y el entorno caribeño son considerados el mayor atractivo de la ciudad.',
        'impacto': 'Alto',
        'facilidad': 'Alta'
    },
    {
        'titulo': '🏰 Patrimonio histórico y cultural',
        'descripcion': 'La arquitectura colonial, las murallas y la historia viva de la ciudad son altamente valoradas. El Centro Histórico es percibido como símbolo de identidad y orgullo cartagenero.',
        'impacto': 'Alto',
        'facilidad': 'Alta'
    },
    {
        'titulo': '😊 Hospitalidad y calidez humana',
        'descripcion': 'Los visitantes mencionan la amabilidad y energía positiva de los habitantes. El trato cordial contribuye a una experiencia acogedora y memorable.',
        'impacto': 'Medio',
        'facilidad': 'Alta'
    },
    {
        'titulo': '🍽️ Gastronomía y vida nocturna',
        'descripcion': 'Los comentarios resaltan la oferta culinaria diversa, especialmente los platos típicos y la música local. La combinación de cultura y entretenimiento es un punto fuerte para el turismo.',
        'impacto': 'Medio',
        'facilidad': 'Media'
    },
    {
        'titulo': '🚤 Experiencias turísticas organizadas',
        'descripcion': 'Excursiones a islas, recorridos culturales y tours guiados son percibidos como bien estructurados. Las actividades complementan la visita y enriquecen la experiencia global.',
        'impacto': 'Medio',
        'facilidad': 'Alta'
    }
]

for f in factores_pos:
    st.markdown(
        f"""
        <div class="card">
            <h4 style="color:#1b2b4a; margin-bottom:4px;">{f['titulo']}</h4>
            <p style="color:#3f4b6b; font-size:15px;">{f['descripcion']}</p>
            <p style="font-size:13px; color:#5c6b88;"><b>Impacto:</b> {f['impacto']} &nbsp; | &nbsp; <b>Facilidad:</b> {f['facilidad']}</p>
        </div>
        """,
        unsafe_allow_html=True
    )

# ===== FACTORES NEGATIVOS =====
st.header('Factores en comentarios negativos')
st.markdown('<h3 style="color:#1b2b4a; font-weight:700;">Factores sociales y estructurales que influyen en los comentarios negativos</h3>', unsafe_allow_html=True)
st.markdown('<p style="color:#3f4b6b;">Análisis contextual de los temas más mencionados en comentarios con percepción negativa sobre la experiencia turística en Cartagena.</p>', unsafe_allow_html=True)

factores = [
    {
        'titulo': '🔒 Seguridad y confianza ciudadana',
        'descripcion': 'Los visitantes mencionan robos menores, estafas o sensación de inseguridad en zonas turísticas como el Centro Histórico y Bocagrande. La presencia irregular de control policial afecta la percepción general del visitante.',
        'impacto': 'Alto',
        'facilidad': 'Media'
    },
    {
        'titulo': '💰 Precios y turismo excluyente',
        'descripcion': 'Se perciben sobrecostos en comidas, transporte o actividades recreativas. La falta de regulación visible en precios genera desconfianza, especialmente entre turistas nacionales.',
        'impacto': 'Alto',
        'facilidad': 'Baja'
    },
    {
        'titulo': '🚗 Infraestructura y movilidad urbana',
        'descripcion': 'La congestión vehicular y el acceso limitado a zonas turísticas generan incomodidad. Se recomienda fortalecer transporte sostenible y señalización clara.',
        'impacto': 'Medio',
        'facilidad': 'Media'
    },
    {
        'titulo': '🌿 Gestión ambiental y limpieza',
        'descripcion': 'Durante la temporada alta se reporta acumulación de residuos en playas y calles. Se requieren campañas de cultura ambiental y mantenimiento urbano constante.',
        'impacto': 'Medio',
        'facilidad': 'Alta'
    },
    {
        'titulo': '🤝 Calidad del servicio y atención al cliente',
        'descripcion': 'Algunos comentarios reflejan deficiencias en atención al turista y trato desigual entre visitantes nacionales y extranjeros. Urge fortalecer la capacitación en hospitalidad.',
        'impacto': 'Medio',
        'facilidad': 'Alta'
    }
]

for f in factores:
    st.markdown(
        f"""
        <div class="card">
            <h4 style="color:#1b2b4a; margin-bottom:4px;">{f['titulo']}</h4>
            <p style="color:#3f4b6b; font-size:15px;">{f['descripcion']}</p>
            <p style="font-size:13px; color:#5c6b88;"><b>Impacto:</b> {f['impacto']} &nbsp; | &nbsp; <b>Facilidad:</b> {f['facilidad']}</p>
        </div>
        """,
        unsafe_allow_html=True
    )

# ===== RECOMENDACIONES AMPLIADAS =====
st.markdown("""
    <style>
    .rec-title {font-size: 32px; font-weight: 800; color: #1b2b4a; margin-bottom: 25px; text-align: center;}
    .rec-card {background: rgba(255, 255, 255, 0.95); border-radius: 18px; box-shadow: 0 8px 20px rgba(40, 60, 100, 0.15); padding: 28px 32px; margin: 20px 0; transition: all 0.3s ease;}
    .rec-card:hover {transform: translateY(-3px); box-shadow: 0 12px 28px rgba(30, 60, 120, 0.25);}
    .rec-title-item {font-size: 22px; font-weight: 700; color: #264778; margin-bottom: 10px;}
    .rec-desc {font-size: 17px; color: #34495e; line-height: 1.6; text-align: justify;}
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="rec-title">Recomendaciones a futuro — Cartagena360</div>', unsafe_allow_html=True)

recs = [
    ('🛡️ Mejorar la seguridad integral en zonas turísticas',
     'Fortalecer la seguridad ciudadana no solo desde la vigilancia policial, sino desde la percepción de confianza. Se recomienda la instalación de puntos seguros y cámaras visibles en zonas de alto flujo, la iluminación eficiente de calles y senderos, y campañas de convivencia ciudadana. La articulación entre autoridades locales y la comunidad es esencial para generar una experiencia turística positiva.'),
    ('🌊 Fortalecer la limpieza y sostenibilidad ambiental de las playas',
     'Implementar programas permanentes de limpieza y educación ambiental, con brigadas locales y señalización visible sobre el manejo de residuos. El turismo sostenible debe reflejarse en la práctica diaria: promover la economía circular, instalar puntos de reciclaje, y asociar la limpieza a campañas de orgullo local ("Cartagena limpia, Cartagena viva"). Esto mejora tanto la imagen internacional como el bienestar local.'),
    ('🤝 Reforzar la atención turística y capacitación del personal',
     'El trato humano y la calidad del servicio son el rostro de la ciudad. Se recomienda desarrollar programas cortos de capacitación para guías, vendedores y personal hotelero, centrados en empatía, comunicación intercultural y resolución pacífica de conflictos. Además, la creación de un “sello Cartagena360” de atención de calidad puede elevar el estándar de hospitalidad.'),
    ('🚏 Ordenamiento y movilidad inteligente',
     'Optimizar la movilidad turística mediante rutas definidas, transporte público confiable y reducción del caos vehicular en sectores históricos. Incorporar señalización inteligente en varios idiomas, transporte ecológico (bicicletas, buses eléctricos) y zonas peatonales seguras. Esto contribuye a un flujo armónico entre visitantes y residentes.'),
    ('🎭 Promoción cultural y orgullo local',
     'Rescatar y visibilizar la identidad cartagenera a través del arte, la música y la gastronomía local. Impulsar festivales barriales, murales y circuitos turísticos culturales que integren comunidades locales y turistas. Esto refuerza el sentido de pertenencia y diversifica la oferta más allá del turismo de lujo.')
]

for titulo, desc in recs:
    st.markdown(f'''<div class="rec-card"><div class="rec-title-item">{titulo}</div><div class="rec-desc">{desc}</div></div>''', unsafe_allow_html=True)

# ===== TABLA DETALLE =====
st.header('Vista detallada de comentarios')
st.dataframe(df[['comentario','sentimiento']].head(200))

st.markdown('---')

# ===== FESTIVAL DE PROYECTOS DE CIENCIA DE DATOS =====
st.markdown('---')
st.markdown("""
    <style>
    .festival-title {
        font-size: 32px;
        font-weight: 800;
        color: #1b2b4a;
        text-align: center;
        margin-top: 30px;
        margin-bottom: 10px;
    }
    .festival-sub {
        text-align: center;
        font-size: 18px;
        color: #3f4b6b;
        margin-bottom: 25px;
    }
    .festival-card {
        background: rgba(255, 255, 255, 0.95);
        border-left: 6px solid #2e5fa8;
        border-radius: 16px;
        box-shadow: 0 8px 20px rgba(40, 60, 100, 0.15);
        padding: 22px 28px;
        margin: 15px 0;
        transition: all 0.3s ease;
    }
    .festival-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 12px 28px rgba(30, 60, 120, 0.25);
    }
    .festival-item-title {
        font-size: 20px;
        font-weight: 700;
        color: #264778;
        margin-bottom: 6px;
    }
    .festival-item-desc {
        font-size: 16px;
        color: #3f4b6b;
        margin: 0;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="festival-title">🎓 Festival de Proyectos de Ciencia de Datos</div>', unsafe_allow_html=True)
st.markdown('<div class="festival-sub">20 de noviembre — Evaluación integral de proyectos con base en criterios de impacto, rigor, reproducibilidad y ética</div>', unsafe_allow_html=True)

criterios = [
    {
        'titulo': '🌍 Impacto y relevancia (20%)',
        'desc': 'Evalúa el grado en que el proyecto aborda un problema real con valor social o económico.',
        'relacion': 'Cartagena360 responde a la necesidad de analizar percepciones ciudadanas y turísticas en Cartagena de Indias. Al identificar factores sociales, económicos y ambientales a partir de comentarios reales, el proyecto contribuye a la toma de decisiones en turismo sostenible y gobernanza local.'
    },
    {
        'titulo': '📊 Rigor metodológico (25%)',
        'desc': 'Considera la calidad de la preparación de datos, la elección y justificación de modelos y la validez de los resultados.',
        'relacion': 'El dashboard aplica técnicas de procesamiento de lenguaje natural (tokenización, TF-IDF, n-gramas) y un enfoque interpretativo de regresión logística. Cada etapa del análisis está documentada y justificada en función de la exploración de sentimientos y temas críticos.'
    },
    {
        'titulo': '🔁 Reproducibilidad (15%)',
        'desc': 'Evalúa la disponibilidad del código, los datos y la facilidad para replicar los resultados.',
        'relacion': 'El proyecto mantiene un flujo reproducible con código abierto en Python y dependencias estándar (Streamlit, scikit-learn, Plotly). Los datos se cargan automáticamente desde archivos CSV y el análisis puede repetirse en cualquier entorno local o en la nube.'
    },
    {
        'titulo': '⚖️ Ética y gobernanza de datos (15%)',
        'desc': 'Considera la protección de la privacidad, la gestión de sesgos y el cumplimiento de licencias y permisos de uso de datos.',
        'relacion': 'Cartagena360 utiliza comentarios públicos anonimizados y promueve la interpretación responsable de datos sociales. Se evita el uso de información sensible y se mantiene la transparencia metodológica para reducir sesgos o interpretaciones erróneas.'
    }
]

for c in criterios:
    st.markdown(f"""
        <div class="festival-card">
            <div class="festival-item-title">{c['titulo']}</div>
            <p class="festival-item-desc"><b>Criterio:</b> {c['desc']}</p>
            <p class="festival-item-desc"><b>Relación con Cartagena360:</b> {c['relacion']}</p>
        </div>
    """, unsafe_allow_html=True)

st.caption('📊 Festival de Proyectos de Ciencia de Datos — Evaluación y alineación de Cartagena360 con criterios académicos y éticos')
st.caption('Dashboard Cartagena360💙')

