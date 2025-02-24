# RAG Technical Assessment for Tailor Hub
## Descripción

Este proyecto implementa un sistema de Recuperación Aumentada con Generación (RAG) que permite transcribir videos de YouTube y Vimeo, almacenar las transcripciones en una base de datos vectorial y realizar consultas sobre los videos a través de un chatbot basado en RAG.

El sistema permite a los usuarios introducir URLs de videos y obtener respuestas relevantes a preguntas sobre su contenido.

## Características principales

1. **Ingesta y procesamiento de datos**
   - Descarga y transcripción de videos de YouTube y Vimeo mediante el modelo Whisper.
2. **Base de datos vectorial**
   - Almacenamiento de las transcripciones en una base de datos vectorial utilizando **ChromaDB**.
   - Generación de embeddings con **OpenAI** para facilitar la búsqueda.
3. **Recuperación de información**
   - Implementación de un mecanismo para recuperar los documentos más relevantes en base a una consulta del usuario.
4. **Generación de respuestas**
   - Integración con **GPT-4o Mini**, seleccionado por ser un modelo eficiente y económico, para generar respuestas basadas en el contexto recuperado.
5. **Frontend interactivo**
   - Implementación de una interfaz gráfica utilizando **Streamlit** para facilitar la interacción con el sistema.

## Requisitos

Para ejecutar el proyecto, es necesario instalar las dependencias alojadas en `requirements.txt`.

### Instalación de dependencias

```bash
pip install -r requirements.txt
```

### Instalación de FFmpeg

Para la correcta transcripción de videos, se requiere **FFmpeg**:

- **Linux (Debian/Ubuntu):**

  ```bash
  sudo apt install ffmpeg
  ```

- **Windows:**
  Descarga e instala desde [FFmpeg](https://ffmpeg.org/download.html).

- **macOS:**

  ```bash
  brew install ffmpeg
  ```

## Uso

Para probar el sistema, clona el repositorio y ejecuta la aplicación con Streamlit:

```bash
git clone https://github.com/SupernovaIa/technical-assessment-tailor-hub.git
cd technical-assessment-tailor-hub
streamlit run Chat.py
```

El repositorio incluye un archivo de ejemplo para pruebas. Este archivo puede eliminarse o sustituirse con otros generados por el usuario.

## Tecnologías utilizadas

- **Python 3.12**
- **ChromaDB** (Base de datos vectorial)
- **FFmpeg** (Procesamiento de audio/video)
- **Streamlit** (Interfaz gráfica)
- **Whisper** (Transcripción de audio)
- **LangChain** (Implementación del RAG y chatbot)
- **OpenAI** (Embeddings)
- **GPT-4o Mini** (Modelo de lenguaje eficiente y económico)

