# ============================================================
# CABECERA
# ============================================================
# Alumno: Raúl Cazorla
# URL Streamlit Cloud: https://spotify-wrappedmda-jbp2rahgvgd5gp8q7ytdba.streamlit.app/
# URL GitHub: https://github.com/rcazorlaluna/Spotify-Wrapped_MDA/

# ============================================================
# IMPORTS
# ============================================================
# Streamlit: framework para crear la interfaz web
# pandas: manipulación de datos tabulares
# plotly: generación de gráficos interactivos
# openai: cliente para comunicarse con la API de OpenAI
# json: para parsear la respuesta del LLM (que llega como texto JSON)
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from openai import OpenAI
import json

# ============================================================
# CONSTANTES
# ============================================================
# Modelo de OpenAI. No lo cambies.
MODEL = "gpt-4.1-mini"

# -------------------------------------------------------
# >>> SYSTEM PROMPT — TU TRABAJO PRINCIPAL ESTÁ AQUÍ <<<
# -------------------------------------------------------
# El system prompt es el conjunto de instrucciones que recibe el LLM
# ANTES de la pregunta del usuario. Define cómo se comporta el modelo:
# qué sabe, qué formato debe usar, y qué hacer con preguntas inesperadas.
#
# Puedes usar estos placeholders entre llaves — se rellenan automáticamente
# con información real del dataset cuando la app arranca:
#   {fecha_min}             → primera fecha del dataset
#   {fecha_max}             → última fecha del dataset
#   {plataformas}           → lista de plataformas (Android, iOS, etc.)
#   {reason_start_values}   → valores posibles de reason_start
#   {reason_end_values}     → valores posibles de reason_end
#
# IMPORTANTE: como el prompt usa llaves para los placeholders,
# si necesitas escribir llaves literales en el texto (por ejemplo para
# mostrar un JSON de ejemplo), usa doble llave: {{ y }}
#
SYSTEM_PROMPT = """
Eres un asistente analítico experto en datos de Spotify. Tu objetivo es generar código en Python para visualizar los datos de escucha del usuario usando la librería Plotly.

Tienes acceso a un DataFrame de pandas llamado `df` con el historial de reproducciones. 
El dataset abarca desde {fecha_min} hasta {fecha_max}.
Las columnas disponibles y limpias son:
- track_name: nombre de la canción (string)
- artist_name: nombre del artista (string)
- album_name: nombre del álbum (string)
- minutes_played: minutos escuchados (float)
- skipped: booleano, True si se saltó la canción, False si no (bool)
- platform: plataforma desde donde se escuchó. Valores posibles: {plataformas}
- reason_start: motivo de inicio. Valores posibles: {reason_start_values}
- reason_end: motivo de fin. Valores posibles: {reason_end_values}
- ts: fecha y hora de la reproducción (datetime)
- year: año de la reproducción (int)
- month: mes de la reproducción (int)
- hour: hora del día (0-23) (int)
- day_name: nombre del día de la semana (string)

REGLAS ESTRICTAS (GUARDRAILS):
1. Si la pregunta se puede responder con estos datos, genera código Python puro que cree una figura usando `px` (plotly.express) o `go` (plotly.graph_objects). Ambas están importadas.
2. Obligatorio: La figura final DEBE asignarse a una variable llamada exactamente `fig`.
3. Prohibido: NO incluyas `fig.show()` en el código.
4. Si la pregunta pide cosas fuera del alcance del dataset (ej. el clima, datos económicos, o buscar canciones en internet), no inventes datos ni generes código.
5. Contexto visual: Si el usuario pregunta por un único elemento (ej. "el artista más escuchado", "mi canción favorita"), no hagas un gráfico de un solo elemento. Amplía automáticamente la visualización a un Top 5 o Top 10 para darle contexto comparativo, aunque en la "interpretacion" respondas a su pregunta singular.

FORMATO DE RESPUESTA OBLIGATORIO:
Debes responder ÚNICAMENTE con un objeto JSON válido, sin texto adicional en markdown. Usa esta estructura exacta:

Si puedes responder:
{{
  "tipo": "grafico",
  "codigo": "fig = px.bar(df, x='...', y='...')",
  "interpretacion": "Mensaje natural y amigable de 1 o 2 líneas explicando al usuario qué muestra el gráfico y dando algún dato curioso si lo hay."
}}

Si la pregunta está fuera de alcance:
{{
  "tipo": "fuera_de_alcance",
  "codigo": "",
  "interpretacion": "Mensaje educado explicando que solo puedes analizar el historial de escucha de Spotify proporcionado."
}}

"""


# ============================================================
# CARGA Y PREPARACIÓN DE DATOS
# ============================================================
# Esta función se ejecuta UNA SOLA VEZ gracias a @st.cache_data.
# Lee el fichero JSON y prepara el DataFrame para que el código
# que genere el LLM sea lo más simple posible.
#
@st.cache_data
def load_data():
    df = pd.read_json("streaming_history.json")

    # ----------------------------------------------------------
    # >>> TU PREPARACIÓN DE DATOS ESTÁ AQUÍ <<<
    # ----------------------------------------------------------
    # Transforma el dataset para facilitar el trabajo del LLM.
    # Lo que hagas aquí determina qué columnas tendrá `df`,
    # y tu system prompt debe describir exactamente esas columnas.
    #
    # Cosas que podrías considerar:
    # - Convertir 'ts' de string a datetime
    # - Crear columnas derivadas (hora, día de la semana, mes...)
    # - Convertir milisegundos a unidades más legibles
    # - Renombrar columnas largas para simplificar el código generado
    # - Filtrar registros que no aportan al análisis (podcasts, etc.)
    # ----------------------------------------------------------

    # 1. Convertimos la columna 'ts' (texto) a un formato de fecha real
    df['ts'] = pd.to_datetime(df['ts'])

    # 2. Extraemos información útil para que el LLM no tenga que calcularla
    df['year'] = df['ts'].dt.year
    df['month'] = df['ts'].dt.month
    df['hour'] = df['ts'].dt.hour
    df['day_name'] = df['ts'].dt.day_name() # Lunes, Martes, etc.

    # 3. Convertimos milisegundos a minutos (mucho más lógico para el análisis)
    df['minutes_played'] = df['ms_played'] / 60000

    # 4. Rellenamos los nulos en 'skipped' (los nulos significan que NO se saltó)
    df['skipped'] = df['skipped'].fillna(False)

    # 5. Renombramos columnas kilométricas para facilitar el código generado
    df = df.rename(columns={
        'master_metadata_track_name': 'track_name',
        'master_metadata_album_artist_name': 'artist_name',
        'master_metadata_album_album_name': 'album_name'
    })

    # 6. Filtramos registros que no aportan (ej. podcasts)
    # En Spotify, los podcasts no tienen 'track_name', así que borramos los nulos de esa columna
    df = df.dropna(subset=['track_name'])
    return df


def build_prompt(df):
    """
    Inyecta información dinámica del dataset en el system prompt.
    Los valores que calcules aquí reemplazan a los placeholders
    {fecha_min}, {fecha_max}, etc. dentro de SYSTEM_PROMPT.

    Si añades columnas nuevas en load_data() y quieres que el LLM
    conozca sus valores posibles, añade aquí el cálculo y un nuevo
    placeholder en SYSTEM_PROMPT.
    """
    fecha_min = df["ts"].min()
    fecha_max = df["ts"].max()
    plataformas = df["platform"].unique().tolist()
    reason_start_values = df["reason_start"].unique().tolist()
    reason_end_values = df["reason_end"].unique().tolist()

    return SYSTEM_PROMPT.format(
        fecha_min=fecha_min,
        fecha_max=fecha_max,
        plataformas=plataformas,
        reason_start_values=reason_start_values,
        reason_end_values=reason_end_values,
    )


# ============================================================
# FUNCIÓN DE LLAMADA A LA API
# ============================================================
# Esta función envía DOS mensajes a la API de OpenAI:
# 1. El system prompt (instrucciones generales para el LLM)
# 2. La pregunta del usuario
#
# El LLM devuelve texto (que debería ser un JSON válido).
# temperature=0.2 hace que las respuestas sean más predecibles.
#
# No modifiques esta función.
#
def get_response(user_msg, system_prompt):
    client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_msg},
        ],
        temperature=0.2,
    )
    return response.choices[0].message.content


# ============================================================
# PARSING DE LA RESPUESTA
# ============================================================
# El LLM devuelve un string que debería ser un JSON con esta forma:
#
#   {"tipo": "grafico",          "codigo": "...", "interpretacion": "..."}
#   {"tipo": "fuera_de_alcance", "codigo": "",    "interpretacion": "..."}
#
# Esta función convierte ese string en un diccionario de Python.
# Si el LLM envuelve el JSON en backticks de markdown (```json...```),
# los limpia antes de parsear.
#
# No modifiques esta función.
#
def parse_response(raw):
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[1] if "\n" in cleaned else cleaned[3:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        cleaned = cleaned.strip()

    return json.loads(cleaned)


# ============================================================
# EJECUCIÓN DEL CÓDIGO GENERADO
# ============================================================
# El LLM genera código Python como texto. Esta función lo ejecuta
# usando exec() y busca la variable `fig` que el código debe crear.
# `fig` debe ser una figura de Plotly (px o go).
#
# El código generado tiene acceso a: df, pd, px, go.
#
# No modifiques esta función.
#
def execute_chart(code, df):
    local_vars = {"df": df, "pd": pd, "px": px, "go": go}
    exec(code, {}, local_vars)
    return local_vars.get("fig")


# ============================================================
# INTERFAZ STREAMLIT
# ============================================================
# Toda la interfaz de usuario. No modifiques esta sección.
#

# Configuración de la página
st.set_page_config(page_title="Spotify Analytics", layout="wide")

# --- Control de acceso ---
# Lee la contraseña de secrets.toml. Si no coincide, no muestra la app.
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if not st.session_state.authenticated:
    st.title("🔒 Acceso restringido")
    pwd = st.text_input("Contraseña:", type="password")
    if pwd:
        if pwd == st.secrets["PASSWORD"]:
            st.session_state.authenticated = True
            st.rerun()
        else:
            st.error("Contraseña incorrecta.")
    st.stop()

# --- App principal ---
st.title("🎵 Spotify Analytics Assistant")
st.caption("Pregunta lo que quieras sobre tus hábitos de escucha")

# Cargar datos y construir el prompt con información del dataset
df = load_data()
system_prompt = build_prompt(df)

# Caja de texto para la pregunta del usuario
if prompt := st.chat_input("Ej: ¿Cuál es mi artista más escuchado?"):

    # Mostrar la pregunta en la interfaz
    with st.chat_message("user"):
        st.write(prompt)

    # Generar y mostrar la respuesta
    with st.chat_message("assistant"):
        with st.spinner("Analizando..."):
            try:
                # 1. Enviar pregunta al LLM
                raw = get_response(prompt, system_prompt)

                # 2. Parsear la respuesta JSON
                parsed = parse_response(raw)

                if parsed["tipo"] == "fuera_de_alcance":
                    # Pregunta fuera de alcance: mostrar solo texto
                    st.write(parsed["interpretacion"])
                else:
                    # Pregunta válida: ejecutar código y mostrar gráfico
                    fig = execute_chart(parsed["codigo"], df)
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)
                        st.write(parsed["interpretacion"])
                        st.code(parsed["codigo"], language="python")
                    else:
                        st.warning("El código no produjo ninguna visualización. Intenta reformular la pregunta.")
                        st.code(parsed["codigo"], language="python")

            except json.JSONDecodeError:
                st.error("No he podido interpretar la respuesta. Intenta reformular la pregunta.")
            except Exception as e:
                st.error("Ha ocurrido un error al generar la visualización. Intenta reformular la pregunta.")


# ============================================================
# REFLEXIÓN TÉCNICA (máximo 30 líneas)
# ============================================================
#
# Responde a estas tres preguntas con tus palabras. Sé concreto
# y haz referencia a tu solución, no a generalidades.
# No superes las 30 líneas en total entre las tres respuestas.
#
# 1. ARQUITECTURA TEXT-TO-CODE
#    ¿Cómo funciona la arquitectura de tu aplicación? ¿Qué recibe
#    el LLM? ¿Qué devuelve? ¿Dónde se ejecuta el código generado?
#    ¿Por qué el LLM no recibe los datos directamente?
#
#    El LLM actúa como un programador, no como un analista de datos. 
#    Recibe las instrucciones (System Prompt) y la estructura de nuestras columnas, y devuelve un texto plano formateado como JSON que contiene código en Python. 
#    Ese código devuelto se ejecuta localmente en nuestro ordenador (con la función exec()).
#    No enviamos los datos directamente por privacidad (no mandamos hábitos íntimos a OpenAI), por coste (enviar 15.000 filas consumiría muchísimos tokens de la API) y por velocidad.

#
#
# 2. EL SYSTEM PROMPT COMO PIEZA CLAVE
#    ¿Qué información le das al LLM y por qué? Pon un ejemplo
#    concreto de una pregunta que funciona gracias a algo específico
#    de tu prompt, y otro de una que falla o fallaría si quitases
#    una instrucción.
#
#    Es el manual de reglas que evita que la IA alucine o rompa la aplicación.
#    Un ejemplo de éxito: Gracias a la regla del "Contexto visual" (la número 5), una pregunta sobre un único artista funciona respondiendo sobre el artista en cuestión pero muestra un Top 10, dándole valor analítico (una gráfica de un solo artista no tiene ningún valor)
#    Ejemplo de fallo: Si quitásemos la regla de "La figura final DEBE asignarse a una variable llamada exactamente fig", el LLM podría llamar al gráfico mi_grafico o plot. Nuestro código local intentaría buscar fig para dibujarlo, no lo encontraría, y la aplicación daría un error técnico.

#
#
# 3. EL FLUJO COMPLETO
#    Describe paso a paso qué ocurre desde que el usuario escribe
#    una pregunta hasta que ve el gráfico en pantalla.
#
#    1. El usuario escribe una pregunta. Por ejemplo: "¿Qué horas escucho más?". 
#    2. La app junta esa pregunta con nuestro System Prompt y se lo envía a la API de OpenAI. 
#    3. OpenAI devuelve un string (texto) con forma de JSON. 
#    4. La app "parsea" ese texto para extraer el código de Python. 
#    5. Tu ordenador ejecuta ese código sobre el df (nuestro dataset local) para generar la imagen. 
#    6. Streamlit pinta esa imagen en la pantalla.
