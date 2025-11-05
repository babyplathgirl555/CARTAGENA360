# # Librerías para dashboard, gráficos y procesamiento de texto
import streamlit as st

import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px

import io
from wordcloud import WordCloud
from sentence_transformers import SentenceTransformer
from sklearn.manifold import TSNE
import utils as utils

# Cargar stopwords personalizadas
STOPWORDS = utils.load_stopwords()

# Rutas de archivos CSV
csv_path = './databases/db_final.csv'
csv_path_old = './databases/twitter_coms.csv'

# Leer bases de datos
df = utils.read_csv_auto(csv_path)
df_old = utils.read_csv_auto(csv_path_old)

# Validación de existencia de archivos
if df is None:
    st.error(f"No se encontró o no se pudo leer el archivo: {csv_path}")
    st.stop()
if df_old is None:
    st.error(f"No se encontró o no se pudo leer el archivo: {csv_path_old}")
    st.stop()

# Personalización de estilo CSS para tarjetas
st.markdown(
    """
    <style>
    /* Tarjetas de factores negativos */
    .card {
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

# Título y descripción del dashboard
st.title("Cartagena 360°: Análisis de Opiniones Turísticas")
st.subheader('Dashboard de Sentimientos Turísticos en Cartagena de Indias')
st.markdown("""
El dashboard fue creado con el objetivo de analizar las opiniones de turistas 
sobre la ciudad de Cartagena de Indias mediante técnicas de procesamiento de 
lenguaje natural y análisis de sentimientos, con el propósito de identificar 
patrones, percepciones y factores determinantes que contribuyan a mejorar la 
experiencia del visitante y fortalecer la competitividad del sector turístico local.
            """)

# Métricas principales
col1, col2, col3 = st.columns(3)
col1.metric('Total de comentarios', len(df))
col2.metric('Promedio longitud', f"{df['comentario'].str.len().mean():.1f}")
col3.metric('Sentimientos únicos', len(df['sentimiento'].unique()))
st.markdown('---')

st.header("Fuente y descripción de los datos")
st.markdown("""
El conjunto de datos utilizado en este proyecto **fue elaborado de manera manual 
por los integrantes del equipo**, a partir de la recopilación de información 
proveniente de plataformas digitales de opinión turística como X.com, TripAdvisor 
y Booking, entre otras.
            """)

# Comparación de bases de datos: viejas vs actualizadas
st.subheader("Comparación de las bases de datos")
col4, col5 = st.columns(2)

with col4:
    st.markdown("#### Base de Datos Actualizada")
    buffer1 = io.StringIO()
    df.iloc[:, :3].info(buf=buffer1)
    st.text(buffer1.getvalue())

with col5:
    st.markdown("#### Base de Datos Original")
    buffer2 = io.StringIO()
    df_old.info(buf=buffer2)
    st.text(buffer2.getvalue())

# Limpieza y depuración de datos
st.subheader("Procesamiento y depuración de los datos")
st.markdown("""
Dado que no todas las fuentes ofrecían de manera consistente la totalidad 
de estas variables, se procedió a depurar el conjunto de datos conservando 
únicamente las columnas más frecuentes y relevantes para el análisis
            
El proceso de limpieza incluyó:
- Conversión de texto a minúsculas y eliminación de espacios innecesarios.
- Eliminación de caracteres no válidos.
- Estandarización de valores categóricos.
- Eliminación de filas con valores nulos en las columnas principales
- Eliminación de registros duplicados.

> El nombre y el usuario son dos formas distintas de identificar la persona 
que escribió el comentario registrado en la base de datos. Sin embargo, ninguno 
de los dos está verdaderamente completo, por lo que se fusionaron los valores
de las dos columnas, priorizando los valores de la columna usuario.
            """)

st.markdown("#### Vista previa de los primeros registros del DataFrame Viejo:")
st.dataframe(df_old.head(10))

st.markdown("#### Vista previa de los primeros registros del DataFrame Actualizado:")
st.dataframe(df.iloc[:, :3].head(10))

st.markdown("""
**Código utilizado**    
```
    # Rellenar NAs en la columna usuario
    df['usuario'] = df['usuario'].fillna(df['nombre'])
            
    df = df.drop('ciudad', axis=1)
    df = df.drop('fecha', axis=1)
    df = df.drop('plataforma', axis=1)
    df = df.drop('nombre', axis=1)
    
    # Limpieza
    columnas_texto = df.select_dtypes(include='object').columns
    print(columnas_texto)

    for col in columnas_texto:
        df[col] = df[col].str.lower().str.strip()
        df[col] = df[col].str.replace(r"[^a-z0-9áéíóúüñ ]", "", regex=True)
    
    # Corrección
    df['pais'] = df['pais'].replace('estados unidos', "usa")
    df['pais'] = df['pais'].replace('brazil', 'brasil')
```
* *Todos estos algoritmos fueron proporcionados en clase*
---
            """)

# Clasificación de comentarios
st.header("Expansión de la base de datos")
col6, col7 = st.columns(2)

with col6:
    st.markdown("#### Clasificación de los comentarios por sentimiento")
    st.markdown("""
    Se utilizó un pipeline de la librería Transformers con el modelo BETO, 
    especializado en análisis de sentimiento para texto en español. Este modelo 
    permite clasificar los comentarios según su polaridad emocional: positivo, 
    negativo o neutral.
        """)

with col7:
    st.markdown("#### Clasificación de los comentarios por contenido")
    st.markdown("""
                Con el propósito de analizar las relaciones semánticas 
                entre los comentarios y detectar posibles similitudes o 
                diferencias en su contenido o tono, se utilizó un modelo 
                de sentence embeddings.

                Finalizando en la aplicación de un algoritmo de agrupamiento 
                mediante DBSCAN, con el fin de identificar conjuntos de 
                comentarios con alto grado de similitud en su contenido o tono 
                emocional. Cada grupo resultante se asignó a una nueva 
                columna denominada “cluster_dbscan”.
                """)

st.markdown("""
**Código utilizado**            
```
    # Agregar columna de sentimiento con BETO
    sentiment = pipeline("sentiment-analysis", model="finiteautomata/beto-sentiment-analysis") 
    df["sentimiento"] = df["comentario"].apply(lambda x: sentiment(x)[0]["label"])
            
    # Agrupamiento de comentarios por contenido
    dbscan = DBSCAN(eps=0.4, min_samples=2, metric="cosine")
    df["cluster_dbscan"] = dbscan.fit_predict(X_emb)
```
""")

st.markdown("#### Vista previa de los primeros registros del DataFrame con sentimientos:")
st.dataframe(df.head(10))
st.markdown("---")

# Vista previa de sentimientos
st.header("Análisis exploratorio de los datos globales")
st.subheader("Análisis de Sentimientos")

# Análisis de sentimientos global
conteo = df["sentimiento"].value_counts()
total = len(df)

col1, col2, col3 = st.columns(3)
col1.metric("🟢 Positivos", conteo.get("pos", 0), f"{conteo.get('pos', 0)/total*100:.1f}%")
col2.metric("🔴 Negativos", conteo.get("neg", 0), f"{conteo.get('neg', 0)/total*100:.1f}%")
col3.metric("⚪ Neutros", conteo.get("neu", 0), f"{conteo.get('neu', 0)/total*100:.1f}%")

# Gráfico de torta de proporción de sentimientos
st.markdown("#### Proporción de Comentarios por Sentimiento")
fig2, ax2 = plt.subplots(figsize=(5,5))
ax2.pie(conteo, labels=conteo.index, autopct='%1.1f%%', colors=["#A8E6CF", "#FF8B94", "#DCD6F7"], startangle=90)
st.pyplot(fig2)


st.subheader("Frecuencia de las palabras")

# Generación de nubes de palabras por sentimiento
def generar_wordcloud(sentimiento, color, sw):
    # Seleccionar los comentarios según el sentimiento
    texto = " ".join(df[df["sentimiento"] == sentimiento]["comentario"])
    
    # Solo generar si hay texto válido
    if texto.strip():
        wc = WordCloud(
            width=1000,
            height=600,
            background_color="white",
            colormap=color,
            stopwords=sw,
            collocations=False
        ).generate(texto)

        # Mostrar la nube directamente en Streamlit
        st.subheader(f"Nube de Palabras - Comentarios {sentimiento.capitalize()}")
        st.image(
            wc.to_array(),
            use_container_width=True
        )

for tipo, color in zip(["pos", "neg", "neu"], ["Greens", "Reds", "Blues"]):
    generar_wordcloud(tipo, color, STOPWORDS)

# Boxplot de longitud de comentarios
st.subheader('Variación entre los comentarios')
fig_len = px.box(df, x='sentimiento', y='longitud')
st.plotly_chart(fig_len, use_container_width=True)

# Embeddings y reducción dimensional para visualización
st.subheader('Comparación entre los comentarios')
modelo = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
X_emb = modelo.encode(df["comentario"], convert_to_tensor=False)
tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)
X_2D = tsne.fit_transform(X_emb)

df_2D = pd.DataFrame(X_2D, columns=["x", "y"])
df_2D["sentimiento"] = df["sentimiento"]
df_2D["pais"] = df["pais"]
df_2D["cluster_dbscan"] = df["cluster_dbscan"]

# Gráfico interactivo de clusters
fig = px.scatter(
    df_2D,
    x="x",
    y="y",
    color="cluster_dbscan",
    hover_data=["sentimiento", "pais"],
    title="Visualización 2D de comentarios (DBSCAN + SentenceTransformer)",
    color_continuous_scale="Viridis"
)

fig.update_traces(marker=dict(size=6, opacity=0.7))

# Nubes de palabras por cluster
st.plotly_chart(fig, use_container_width=True)

col8, col9 = st.columns(2)
for c, cluster in enumerate(sorted(df["cluster_dbscan"].unique())):
    textos = df.loc[df["cluster_dbscan"] == cluster, "comentario"]
    texto = " ".join(textos)
    wc = WordCloud(
        width=1000,
        height=600,
        background_color="white",
        colormap="viridis",
        stopwords=STOPWORDS,
        collocations=False
    ).generate(texto)

    if (c + 1) % 2 == 0:
        with col9:
            st.markdown(f"### Nube de Palabras - Cluster {cluster}")
            st.image(wc.to_array(), use_container_width=True)
    else:
        with col8:
            st.markdown(f"### Nube de Palabras - Cluster {cluster}")
            st.image(wc.to_array(), use_container_width=True)

# Análisis por países: nacional vs exterior
st.header("Análisis exploratorio de los datos por países")
df_extended = df
df_extended["origen"] = df_extended["pais"].apply(
    lambda x: "Nacional" if str(x).strip().lower() == "colombia" else "Exterior"
)

# Calcular distribución de sentimientos por origen ---
conteo_sent = (
    df_extended.groupby(["origen", "sentimiento"])
    .size()
    .reset_index(name="cuenta")
)


conteo_sent["sentimiento"] = conteo_sent["sentimiento"].str.upper().str.strip()

col4, col5 = st.columns(2)
with col4:
    st.markdown("#### Sentimientos - Nacional")
    nacional = conteo_sent[conteo_sent["origen"] == "Nacional"]
    if not nacional.empty:
        fig1, ax1 = plt.subplots(figsize=(5, 5))
        ax1.pie(
            nacional["cuenta"],
            labels=nacional["sentimiento"],
            autopct="%1.1f%%",
            startangle=90,
            colors=["#A8E6CF", "#FF8B94", "#DCD6F7"],  # POS, NEU, NEG
        )
        ax1.axis("equal")
        st.pyplot(fig1, use_container_width=True)
    else:
        st.info("No hay datos de origen Nacional.")
with col5:
    st.markdown("#### Sentimientos - Exterior")
    exterior = conteo_sent[conteo_sent["origen"] == "Exterior"]
    if not exterior.empty:
        fig2, ax2 = plt.subplots(figsize=(5, 5))
        ax2.pie(
            exterior["cuenta"],
            labels=exterior["sentimiento"],
            autopct="%1.1f%%",
            startangle=90,
            colors=["#A8E6CF", "#FF8B94", "#DCD6F7"],
        )
        ax2.axis("equal")
        st.pyplot(fig2, use_container_width=True)
    else:
        st.info("No hay datos de origen Exterior.")

pais_sentimiento = df.groupby(["pais", "sentimiento"]).size().reset_index(name="cantidad")
fig = px.bar(
    pais_sentimiento,
    x="pais",
    y="cantidad",
    color="sentimiento",
    barmode="group",
    title="Distribución de sentimientos por país",
    color_discrete_map={"POS": "green", "NEU": "gray", "NEG": "red"},
    labels={
        "pais": "País",
        "cantidad": "Cantidad de comentarios",
        "sentimiento": "Sentimiento"
    }
)

fig.update_layout(
    xaxis_tickangle=-45,
    yaxis_title="Cantidad de comentarios",
    xaxis_title="País"
)

st.plotly_chart(fig, use_container_width=True)  


# Mapeo numérico de sentimientos y gráfico apilado por porcentaje
sent_map = {"POS": 1, "NEU": 0, "NEG": -1}
df["sentimiento"] = df["sentimiento"].str.upper().str.strip()
df["sentimiento_valor"] = df["sentimiento"].map(sent_map)
df_bar = (
    df.groupby(["pais", "sentimiento"])
      .size()
      .reset_index(name="cuenta")
)
df_total = df_bar.groupby("pais")["cuenta"].transform("sum")
df_bar["porcentaje"] = df_bar["cuenta"] / df_total * 100

fig = px.bar(
    df_bar,
    x="pais",
    y="porcentaje",
    color="sentimiento",
    color_discrete_map={"pos": "green", "neu": "gray", "neg": "red"},
    title="Porcentaje de sentimientos por país",
    text="porcentaje",
    barmode="stack"
)

fig.update_traces(
    texttemplate="%{text:.1f}%",
    textposition="inside"
)
fig.update_layout(
    xaxis_tickangle=-45,
    yaxis_title="Porcentaje (%)",
    xaxis_title="País",
    legend_title="Sentimiento",
    uniformtext_minsize=8,
    uniformtext_mode="hide"
)

st.plotly_chart(fig, use_container_width=True)

st.markdown('---')

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

st.caption('Festival de Proyectos de Ciencia de Datos — Evaluación y alineación de Cartagena360 con criterios académicos y éticos')
st.caption('Dashboard Cartagena360💙')