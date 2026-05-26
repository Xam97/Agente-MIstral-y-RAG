# 📊 Sales Bot con LangChain + Mistral + RAG

Este proyecto es un bot inteligente para analizar datos de ventas usando IA.  
Permite hacer preguntas en lenguaje natural sobre un archivo CSV utilizando:

- LangChain
- Mistral AI
- Pandas
- RAG (Retrieval-Augmented Generation)
- FAISS

---

# 🚀 Funciones principales

✅ Cargar archivos CSV  
✅ Limpiar y normalizar datos  
✅ Detectar duplicados con fuzzy matching  
✅ Crear métricas automáticamente  
✅ Usar búsqueda semántica con RAG  
✅ Responder preguntas sobre ventas en español  

---

# 🧠 Tecnologías usadas

- Python
- Pandas
- LangChain
- Mistral AI
- FAISS
- Sentence Transformers
- TheFuzz
- Google Colab

---

# 📦 Instalación

```bash
pip install langchain langchain-community langchain-experimental \
langchain-mistralai pandas openpyxl thefuzz python-Levenshtein \
mistralai faiss-cpu sentence-transformers
```

---

# 🔑 Configuración API Key

El sistema usa una API Key de Mistral AI.

Podés obtener una en:

https://mistral.ai/

En Colab:

```python
from google.colab import userdata
MISTRAL_API_KEY = userdata.get('MISTRAL_API_KEY')
```

O manualmente:

```python
MISTRAL_API_KEY = input("Pegá tu API KEY: ")
```

---

# ▶️ Cómo usar

1. Ejecutar todas las celdas del notebook
2. Subir un archivo CSV
3. Escribir preguntas en el chat

Ejemplos:

```text
¿Cuál es el país con más ventas?
```

```text
Calcula el top 5 de clientes por ventas
```

```text
¿Cuál es el promedio de ventas?
```

---

# 🔍 ¿Qué hace el RAG?

El sistema RAG:

1. Convierte datos en documentos
2. Genera embeddings
3. Guarda vectores en FAISS
4. Busca información relevante
5. Mejora las respuestas del modelo

---

# 📊 Comandos disponibles

| Comando | Función |
|---|---|
| info | Ver información del dataset |
| rag_on | Activar RAG |
| rag_off | Desactivar RAG |
| estado | Ver estado del sistema |
| salir | Cerrar el bot |

---

# 📁 Estructura del proyecto

```text
Agente.ipynb
```

---

# ⚠️ Recomendaciones

- Usar datasets menores a 5000 filas para activar RAG
- Tener conexión a internet
- Usar archivos CSV de ventas

---

# 👨‍💻 Autor

Proyecto académico desarrollado con Python, LangChain y Mistral AI.
