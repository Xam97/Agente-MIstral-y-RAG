# Sistema Multiagente para Análisis de Sentimientos con RAG

## Descripción

Proyecto desarrollado en Python y Google Colab para el análisis de sentimientos sobre el dataset Amazon Fine Food Reviews. El sistema utiliza una arquitectura basada en tres agentes inteligentes que realizan la limpieza de datos, el entrenamiento de modelos y la generación de respuestas mediante RAG y Mistral.

## Arquitectura

### Agente 1: Normalizador

* Limpieza de texto.
* Eliminación de duplicados y valores nulos.
* Balanceo de clases.
* Generación del dataset final para entrenamiento.

### Agente 2: Comparador de Modelos

Entrena y evalúa tres modelos:

* Regresión Logística
* LSTM
* DistilBERT

Las métricas utilizadas son:

* Accuracy
* Precision
* Recall
* F1-Score

El agente selecciona automáticamente el modelo con mejor rendimiento.

### Agente 3: Comunicador

Implementa un sistema RAG utilizando:

* ChromaDB
* Mistral AI

Permite responder preguntas sobre las reseñas y generar reportes del proyecto.

## Tecnologías Utilizadas

* Python
* Pandas
* NumPy
* Scikit-Learn
* TensorFlow / Keras
* Transformers
* DistilBERT
* ChromaDB
* Mistral AI

## Dataset

Amazon Fine Food Reviews

* Más de 568.000 reseñas de productos.
* Datos obtenidos desde Kaggle.
* Clasificación de sentimientos positivos y negativos.

## Instalación

```bash
pip install pandas numpy matplotlib seaborn
pip install scikit-learn tensorflow keras
pip install transformers torch
pip install sentence-transformers chromadb
pip install langchain-mistralai
```

## Configuración

Configurar las siguientes credenciales en Google Colab:

```python
KAGGLE_USERNAME
KAGGLE_KEY
MISTRAL_API_KEY
```

## Ejecución

Ejecutar los bloques del notebook en orden:

1. Instalación de dependencias.
2. Importaciones.
3. Configuración inicial.
4. Descarga del dataset.
5. Agente 1: Normalización.
6. Preparación de datos.
7. Agente 2: Comparación de modelos.
8. Creación del corpus RAG.
9. Configuración de Mistral.
10. Agente 3: Comunicador.
11. Ejecución final.

## Funcionalidades

* Limpieza y preparación de datos.
* Clasificación automática de sentimientos.
* Comparación de modelos de IA.
* Recuperación de contexto mediante RAG.
* Respuestas generadas con Mistral.
* Consulta interactiva sobre el dataset.
* Generación automática de reportes.

