# -*- coding: utf-8 -*-


# FASE 0: PREPARACIÓN DEL ENTORNO
"""

print("\n" + "═" * 70)
print("CONFIGURANDO EL TALLER DE DATOS (VERSION CORREGIDA)")
print("═" * 70 + "\n")

# Instalación de herramientas necesarias
!pip install -q pandas numpy scikit-learn matplotlib seaborn wordcloud openpyxl pyarrow

from google.colab import drive
drive.mount('/content/drive')

# Creando el espacio de trabajo
!mkdir -p /content/drive/MyDrive/steam_agents_project

import pandas as pd
import numpy as np
import re
import json
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

print("Entorno listo para trabajar")
print(f"Sesión iniciada: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

"""# FASE 1: CARGANDO LOS DATOS"""

print("═" * 70)
print("📥 IMPORTANDO EL CATÁLOGO DE STEAM")
print("═" * 70 + "\n")

DATA_PATH = '/content/drive/MyDrive/steam_games.csv'
raw_dataframe = pd.read_csv(DATA_PATH)

print(f"Dataset importado con éxito")
print(f"Formato: {raw_dataframe.shape[0]} juegos × {raw_dataframe.shape[1]} atributos")
print(f"Columnas disponibles: {', '.join(raw_dataframe.columns[:5])}...")
print("\nVista previa:")
display(raw_dataframe.head())

"""# FASE 2: DIAGNÓSTICO DEL PACIENTE (EDA)"""

print("\n" + "═" * 70)
print("DIAGNOSTICANDO LA SALUD DE LOS DATOS")
print("═" * 70 + "\n")

def diagnosticar_dataset(df):
    """Realiza un chequeo completo del estado del dataset"""
    print("1. TIPOS DE DATOS")
    print(f"   → {dict(df.dtypes.value_counts())}\n")

    print("2. VALORES AUSENTES (Top 5)")
    nulos = df.isnull().sum()
    nulos_pct = (nulos / len(df)) * 100
    reporte = pd.DataFrame({'Ausentes': nulos, '%': nulos_pct})
    print(reporte[reporte['Ausentes'] > 0].sort_values('Ausentes', ascending=False).head(5).to_string())

    print("\n3. ESTADÍSTICAS DE COLUMNAS NUMÉRICAS")
    print(df.describe().to_string())

    print("\n4. MUESTRAS DE CAMPOS CRÍTICOS")
    campos_especiales = ['original_price', 'recent_reviews', 'release_date', 'genre']
    for campo in campos_especiales:
        if campo in df.columns:
            print(f"\n   → {campo}:")
            print(f"      {df[campo].dropna().head(3).tolist()}")

diagnorstico = diagnosticar_dataset(raw_dataframe)

"""# FASE 3: AGENTE LIMPIADOR (Clase principal)"""

print("\n" + "═" * 70)
print("ACTIVANDO AL AGENTE LIMPIADOR REFINADO")
print("═" * 70 + "\n")

class DataCleaningAgent:
    """
    🧹 AGENTE DE LIMPIEZA ESPECIALIZADO EN DATOS DE STEAM
    - Resuelve conflictos bilingües en fechas (Español / Inglés).
    - Limpia strings monetarios complejos con símbolos de divisa ($).
    - Extrae métricas cuantitativas y cualitativas de reseñas.
    - Prepara estructuras para analítica avanzada y ML.
    """

    def __init__(self, dataframe):
        self.datos_originales = dataframe.copy()
        self.datos_procesados = dataframe.copy()
        self.bitacora = []

    def registrar(self, mensaje):
        """Registra cada acción en la bitácora"""
        momento = datetime.now().strftime("%H:%M:%S")
        entrada = f"[{momento}] {mensaje}"
        self.bitacora.append(entrada)
        print(f"   ✓ {mensaje}")

    def estandarizar_precios(self):
        """Convierte precios de texto a números flotantes manejando símbolos de moneda"""
        self.registrar("Normalizando precios y removiendo símbolos ($)...")

        def extraer_valor_monetario(valor):
            if pd.isna(valor):
                return np.nan

            texto = str(valor).lower().strip()

            # Detectando juegos gratuitos de forma expandida
            if any(palabra in texto for palabra in ['free', 'gratis', 'demo', 'free to play']):
                return 0.0

            # CORRECCIÓN: Limpieza de caracteres de moneda comunes para evitar fallas en la regex
            texto = texto.replace('$', '').replace('€', '').strip()

            # Extrayendo el número
            coincidencia = re.search(r'(\d+[\.,]\d{2})|(\d+)', texto)
            if coincidencia:
                numero = coincidencia.group(0).replace(',', '.')
                try:
                    return float(numero)
                except ValueError:
                    return np.nan
            return np.nan

        self.datos_procesados['precio_limpio'] = self.datos_procesados['original_price'].apply(extraer_valor_monetario)

        if 'discount_price' in self.datos_procesados.columns:
            self.datos_procesados['descuento_limpio'] = self.datos_procesados['discount_price'].apply(extraer_valor_monetario)
            # Asegurar que el descuento vacío herede el precio base original limpio
            self.datos_procesados['descuento_limpio'] = self.datos_procesados['descuento_limpio'].fillna(self.datos_procesados['precio_limpio'])

            # Calculando porcentaje de ahorro real
            mascara = (self.datos_procesados['descuento_limpio'] < self.datos_procesados['precio_limpio']) & (self.datos_procesados['precio_limpio'] > 0)
            self.datos_procesados['porcentaje_descuento'] = np.where(
                mascara,
                100 * (1 - self.datos_procesados['descuento_limpio'] / self.datos_procesados['precio_limpio']),
                0
            )
        return self

    def procesar_resenas(self):
        """Extrae métricas de reseñas en columnas separadas limpiando strings estructurados"""
        self.registrar("Extrayendo métricas cuantitativas y cualitativas de reseñas...")

        def analizar_resena(texto):
            if pd.isna(texto):
                return {'sentimiento': 'Sin reseñas', 'cantidad': 0, 'porcentaje': 0}

            texto = str(texto).strip()

            # Captura del sentimiento ignorando guiones o comas iniciales
            sentimiento_match = re.match(r'^([^,\-]+)', texto)
            sentimiento = sentimiento_match.group(1).strip() if sentimiento_match else 'Desconocido'

            # Extracción del volumen total de reviews
            cantidad_match = re.search(r'\(([\d,]+)\)', texto)
            cantidad = int(cantidad_match.group(1).replace(',', '')) if cantidad_match else 0

            # Extracción de la tasa de aprobación porcentual
            porcentaje_match = re.search(r'(\d+)%', texto)
            porcentaje = float(porcentaje_match.group(1)) if porcentaje_match else 0

            return {'sentimiento': sentimiento, 'cantidad': cantidad, 'porcentaje': porcentaje}

        for columna in ['recent_reviews', 'all_reviews']:
            if columna in self.datos_procesados.columns:
                datos_analizados = self.datos_procesados[columna].apply(analizar_resena)
                self.datos_procesados[f'{columna}_sentimiento'] = datos_analizados.apply(lambda x: x['sentimiento'])
                self.datos_procesados[f'{columna}_cantidad'] = datos_analizados.apply(lambda x: x['cantidad'])
                self.datos_procesados[f'{columna}_porcentaje'] = datos_analizados.apply(lambda x: x['porcentaje'])
        return self

    def procesar_fechas(self):
        """Convierte fechas resolviendo la inconsistencia bilingüe de meses"""
        self.registrar("Estandarizando fechas bilingües (ES / EN)...")

        def limpiar_fecha_texto(txt):
            if pd.isna(txt):
                return np.nan
            txt = str(txt).lower().strip()

            # Diccionario para mapear abreviaciones en español al estándar internacional legible por pandas
            mapeo_meses = {
                'ene.': 'Jan', 'feb.': 'Feb', 'mar.': 'Mar', 'abr.': 'Apr',
                'may.': 'May', 'jun.': 'Jun', 'jul.': 'Jul', 'ago.': 'Aug',
                'sep.': 'Sep', 'oct.': 'Oct', 'nov.': 'Nov', 'dic.': 'Dec'
            }
            for esp, ing in mapeo_meses.items():
                txt = txt.replace(esp, ing)
            return txt

        fechas_traducidas = self.datos_procesados['release_date'].apply(limpiar_fecha_texto)

        self.datos_procesados['fecha_lanzamiento'] = pd.to_datetime(
            fechas_traducidas,
            format='mixed',
            errors='coerce'
        )

        # Extracción de features temporales útiles para Machine Learning
        self.datos_procesados['año_lanzamiento'] = self.datos_procesados['fecha_lanzamiento'].dt.year
        self.datos_procesados['mes_lanzamiento'] = self.datos_procesados['fecha_lanzamiento'].dt.month
        self.datos_procesados['trimestre'] = self.datos_procesados['fecha_lanzamiento'].dt.quarter
        self.datos_procesados['dia_semana'] = self.datos_procesados['fecha_lanzamiento'].dt.dayofweek
        self.datos_procesados['lanzamiento_fin_semana'] = (self.datos_procesados['dia_semana'] >= 5).astype(float)
        self.datos_procesados['juego_clasico'] = (self.datos_procesados['año_lanzamiento'] < 2010).astype(float)
        return self

    def procesar_listas(self, nombre_columna):
        """Convierte strings estructurados de listas en arrays reales de Python"""
        if nombre_columna not in self.datos_procesados.columns:
            return self

        self.registrar(f"Parseando columna de lista: {nombre_columna}")

        def convertir_lista(valor):
            if pd.isna(valor):
                return []
            texto = str(valor).strip()
            if texto.startswith('[') and texto.endswith(']'):
                try:
                    return json.loads(texto.replace("'", '"'))
                except:
                    return [x.strip().replace("'", "").replace('"', '') for x in texto[1:-1].split(',')]
            else:
                return [x.strip() for x in texto.split(',') if x.strip()]

        self.datos_procesados[f'{nombre_columna}_lista'] = self.datos_procesados[nombre_columna].apply(convertir_lista)
        self.datos_procesados[f'{nombre_columna}_cantidad'] = self.datos_procesados[f'{nombre_columna}_lista'].apply(len)
        return self

    def rellenar_vacios(self):
        """Imputa valores faltantes basándose en el tipo de columna de forma segura"""
        self.registrar("Imputando valores ausentes...")

        # Columnas de texto categórico
        texto_cols = ['developer', 'publisher', 'recent_reviews_sentimiento', 'all_reviews_sentimiento']
        for col in texto_cols:
            if col in self.datos_procesados.columns:
                self.datos_procesados[col] = self.datos_procesados[col].fillna('Desconocido')

        # Descripciones largas
        desc_cols = ['desc_snippet', 'about_the_game']
        for col in desc_cols:
            if col in self.datos_procesados.columns:
                self.datos_procesados[col] = self.datos_procesados[col].fillna('')

        # Columnas numéricas (usando mediana para evitar distorsiones por outliers)
        numericas = self.datos_procesados.select_dtypes(include=[np.number]).columns
        for col in numericas:
            if self.datos_procesados[col].isnull().any():
                mediana = self.datos_procesados[col].median()
                self.datos_procesados[col] = self.datos_procesados[col].fillna(mediana if not pd.isna(mediana) else 0)
        return self

    def preparar_objetivo(self):
        """Genera una métrica logarítmica ponderada estable para predecir (Score)"""
        self.registrar("Construyendo variable objetivo ponderada...")

        if 'all_reviews_porcentaje' in self.datos_procesados.columns and 'all_reviews_cantidad' in self.datos_procesados.columns:
            pct = self.datos_procesados['all_reviews_porcentaje'].fillna(0)
            cnt = self.datos_procesados['all_reviews_cantidad'].fillna(0)

            # Castigo logarítmico para juegos con poquísimos votos
            self.datos_procesados['puntaje_juego'] = pct * np.log1p(cnt)
            max_score = self.datos_procesados['puntaje_juego'].max()

            if max_score > 0:
                self.datos_procesados['puntaje_normalizado'] = 100 * self.datos_procesados['puntaje_juego'] / max_score
            else:
                self.datos_procesados['puntaje_normalizado'] = 0
        return self

    def obtener_resultado(self):
        """Retorna el DataFrame limpio junto al log de auditoría"""
        return self.datos_procesados, self.bitacora

    def archivar_datos(self, ruta):
        """Exporta el dataset serializando listas internas para compatibilidad completa"""
        df_para_guardar = self.datos_procesados.copy()
        for col in df_para_guardar.columns:
            if col.endswith('_lista'):
                df_para_guardar[col] = df_para_guardar[col].apply(json.dumps)

        # Guardar en Parquet (Optimizado)
        df_para_guardar.to_parquet(ruta, index=False)
        self.registrar(f"Datos guardados exitosamente en Parquet: {ruta}")

        # Guardar logs en JSON
        log_path = ruta.replace('.parquet', '_bitacora.json')
        with open(log_path, 'w') as f:
            json.dump(self.bitacora, f, indent=2)
        return self

"""# FASE 4: EJECUTANDO EL PROCESAMIENTO"""

print("═" * 70)
print("🏭 PROCESANDO Y PULIENDO LOS DATOS CON EL PIPELINE")
print("═" * 70 + "\n")

# Instancia del agente con correcciones integradas
agente_limpiador = DataCleaningAgent(raw_dataframe)

# Ejecución secuencial fluida del Pipeline
agente_limpiador \
    .estandarizar_precios() \
    .procesar_resenas() \
    .procesar_fechas() \
    .procesar_listas('genre') \
    .procesar_listas('popular_tags') \
    .procesar_listas('languages') \
    .rellenar_vacios() \
    .preparar_objetivo()

# Extracción de variables limpias finales
datos_limpios, historial = agente_limpiador.obtener_resultado()

print("\n" + "═" * 70)
print("PROCESAMIENTO FINALIZADO")
print("═" * 70)
print(f"\nResumen del dataset procesado:")
print(f"   → Registros únicos de juegos: {len(datos_limpios):,}")
print(f"   → Atributos/Features finales: {len(datos_limpios.columns)}")
print(f"   → Operaciones del agente registradas: {len(historial)}")
print(f"   → Consumo de memoria ram: {datos_limpios.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

print("\nMuestra del resultado final:")
display(datos_limpios.head())

"""# FASE 5: RESPALDANDO EL RESULTADO"""

print("\n" + "═" * 70)
print("RESPALDANDO DATOS EN GOOGLE DRIVE")
print("═" * 70 + "\n")

RUTA_BASE = '/content/drive/MyDrive/steam_agents_project/'
archivo_final_parquet = RUTA_BASE + 'steam_games_procesado.parquet'
archivo_final_csv = RUTA_BASE + 'steam_games_procesado.csv'

# Guardar logs y dataframes procesados
agente_limpiador.archivar_datos(archivo_final_parquet)
datos_limpios.to_csv(archivo_final_csv, index=False)

print(f"\nDirectorio de guardado: {RUTA_BASE}")
print(f"   → [OK] Parquet Engine: steam_games_procesado.parquet")
print(f"   → [OK] CSV Engine: steam_games_procesado.csv")
print(f"   → [OK] Log Audit: steam_games_procesado_bitacora.json")

"""# FASE 6: VISUALIZANDO LOS RESULTADOS"""

print("\n" + "═" * 70)
print("CONFIGURANDO DASHBOARD ESTÁTICO DE CONTROL")
print("═" * 70 + "\n")

# Ajuste automático del tema visual seguro para Colab
plt.style.use('seaborn-v0_8-darkgrid')

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 11))
fig.suptitle('DASHBOARD ANALÍTICO DE JUEGOS - STEAM DATASET', fontsize=16, fontweight='bold', color='#1b2838')

# Gráfico 1: Histograma de precios corregido
precios_filtrados = datos_limpios[datos_limpios['precio_limpio'] < 100]['precio_limpio'].dropna()
ax1.hist(precios_filtrados, bins=40, edgecolor='#121a24', alpha=0.8, color='#107c11')
ax1.set_title('Distribución de Precios Base (< $100)', fontsize=12, fontweight='bold')
ax1.set_xlabel('Precio de venta original (USD)')
ax1.set_ylabel('Cantidad de Títulos')
ax1.axvline(precios_filtrados.median(), color='#e81123', linestyle='--', linewidth=2, label=f'Mediana: ${precios_filtrados.median():.2f}')
ax1.legend()

# Gráfico 2: Distribución de la métrica objetivo ponderada
if 'puntaje_normalizado' in datos_limpios.columns:
    ax2.hist(datos_limpios['puntaje_normalizado'].dropna(), bins=30, edgecolor='#121a24', alpha=0.8, color='#0078d4')
    ax2.set_title('Distribución de Puntaje Objetivo Ponderado', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Score Normalizado Ponderado por Volumen (0-100)')
    ax2.set_ylabel('Frecuencia de Juegos')

# Gráfico 3: Línea de lanzamientos anuales históricos (Filtrado temporal desde 1995)
conteo_anual = datos_limpios['año_lanzamiento'].value_counts().sort_index()
conteo_anual_filtrado = conteo_anual[conteo_anual.index >= 1995]
ax3.bar(conteo_anual_filtrado.index, conteo_anual_filtrado.values, alpha=0.8, color='#ff8c00', edgecolor='#121a24')
ax3.set_title('Evolución de Lanzamientos por Año', fontsize=12, fontweight='bold')
ax3.set_xlabel('Año del Lanzamiento Oficial')
ax3.set_ylabel('Volumen de Juegos Publicados')
ax3.set_xticks(conteo_anual_filtrado.index[::2]) # Mostrar marcas de año cada dos posiciones para evitar encimamiento
ax3.tick_params(axis='x', rotation=45)

# Gráfico 4: Análisis de Frecuencia de los 10 Géneros Dominantes
if 'genre_list' in datos_limpios.columns:
    todos_los_generos = []
    for lista in datos_limpios['genre_list'].dropna():
        todos_los_generos.extend(lista)

    top_generos = dict(Counter(todos_los_generos).most_common(10))
    g_colores = plt.cm.Blues(np.linspace(0.4, 0.9, 10))
    ax4.barh(list(top_generos.keys()), list(top_generos.values()), color=g_colores, edgecolor='#121a24')
    ax4.set_title('🎮 Top 10 Géneros más Concurrentes', fontsize=12, fontweight='bold')
    ax4.set_xlabel('Presencia Total en Juegos')
    ax4.invert_yaxis()

plt.tight_layout()
plt.savefig(RUTA_BASE + 'dashboard_steam.png', dpi=150, bbox_inches='tight')
plt.show()

print(f"Dashboard gráfico guardado correctamente en: {RUTA_BASE}dashboard_steam.png")
print("\n" + "═" * 70)
print("¡TODO EL PROCESO FINALIZÓ EXITOSAMENTE!")
print("═" * 70 + "\n")