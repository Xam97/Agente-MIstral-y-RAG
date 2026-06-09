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

