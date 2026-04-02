# 🎵 Spotify Analytics Assistant (Text-to-Code)

Este repositorio contiene la solución para el **Business Case 5 del máster MDA**. 

## 📌 Sobre el proyecto
Se trata de un asistente analítico conversacional construido con Python y Streamlit. Permite al usuario explorar su historial real de escuchas de Spotify (exportado en formato JSON) realizando preguntas directas en lenguaje natural.

La aplicación utiliza una arquitectura **Text-to-Code**: el modelo de lenguaje (OpenAI) no tiene acceso a los datos brutos por motivos de privacidad y eficiencia. En su lugar, recibe la estructura del dataset a través de un *System Prompt* fuertemente parametrizado y genera código en Python que se ejecuta en el entorno local para devolver visualizaciones interactivas al usuario.

## 🛠️ Tecnologías utilizadas
* **Interfaz y Backend:** Streamlit
* **Manipulación de datos:** Pandas
* **Visualización:** Plotly (Express y Graph Objects)
* **Inteligencia Artificial:** API de OpenAI (gpt-4.1-mini)
