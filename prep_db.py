from utils import read_csv_auto

URL_DATA = './twitter_coms.csv'

df = read_csv_auto(URL_DATA)

# Rellenar valores faltantes en 'usuario' usando 'nombre'
df['usuario'] = df['usuario'].fillna(df['nombre'])
        
# Eliminar columnas innecesarias
df_no_city = df.drop('ciudad', axis=1)
df_no_date = df_no_city.drop('fecha', axis=1)
df_no_platform = df_no_date.drop('plataforma', axis=1)
df_no_name = df_no_platform.drop('nombre', axis=1)

strip_df = df_no_name.copy()

# Columnas de texto a limpiar
columnas_texto = strip_df.select_dtypes(include='object').columns
print(columnas_texto)

# Limpieza de texto: minúsculas, espacios, caracteres especiales
for col in columnas_texto:
    strip_df[col] = strip_df[col].str.lower().str.strip()
    strip_df[col] = strip_df[col].str.replace(r"[^a-z0-9áéíóúüñ ]", "", regex=True)

clean_df = strip_df.copy()

# Normalización de nombres de países
clean_df['pais'] = clean_df['pais'].replace('estados unidos', "usa")
clean_df['pais'] = clean_df['pais'].replace('brazil', 'brasil')

clean_df.to_csv('./db_final.csv', index=False)
