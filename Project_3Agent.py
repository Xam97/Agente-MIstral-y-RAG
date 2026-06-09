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

