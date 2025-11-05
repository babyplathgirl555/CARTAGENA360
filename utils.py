# Imports necesarios para procesamiento de texto, manejo de datos y detección de codificación
import nltk
from nltk.corpus import stopwords

import pandas as pd
import chardet
import os


def load_stopwords():
    # Verificar si los stopwords de NLTK están disponibles, si no, descargarlos
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')

    # Crear conjunto de stopwords en español y agregar palabras frecuentes irrelevantes para análisis
    stopwords_es = set(stopwords.words('spanish'))
    stopwords_es.update([
        "playa", "blanca", "cartagena", "gracias", "pues", "muy", "cómo",
        "más", "menos", "ser", "estar", "tener", "hacer", "ir", "aquí","que","los",
        "con", "por", "para", "una","las","pero","del","todo","hay","fue","nos",
        "como", "gente", "les", "guía", "mucho","tour","mejor","eso","día","era","son",
        "qué","solo","porque","cuando","tiene","cada","santa","este","playas","vale",
        "así","ciudad","colombia","tan","esta","está","lugares","todos", "si"
    ])

    return stopwords_es


def read_csv_auto(filename):
    # Retornar None si el archivo no existe
    if not os.path.exists(filename):
        return None

    # Detectar codificación automáticamente
    with open(filename, 'rb') as f:
        enc = chardet.detect(f.read())['encoding']

    # Intentar leer el CSV con distintos separadores comunes
    for sep in [',', ';', '\t']:
        try:
            df_temp = pd.read_csv(filename, sep=sep, encoding=enc, on_bad_lines='skip')
            # Validar que tenga más de una columna para considerar que la lectura fue correcta
            if df_temp.shape[1] > 1:
                return df_temp
        except Exception:
            continue

    return None
