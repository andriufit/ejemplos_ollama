#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=============================================================================
 PRUEBA DE OLLAMA CLOUD (modelo potente)  -  Material didáctico
=============================================================================
 Objetivo: usar un modelo grande alojado en la nube de Ollama (gpt-oss:120b),
 con TODOS los parámetros de la llamada, leyendo la API KEY de un archivo
 externo para no mostrarla en clase.
 
 Contraste con el script local:
   - Local (gemma3:4b): NO admite tools nativas -> function calling por prompt.
   - Cloud (gpt-oss:120b): SÍ admite tools nativas -> usamos la API 'tools='.
 
 Requisitos previos:
   1) Cuenta en ollama.com y una API key creada en:  ollama.com/settings/keys
   2) pip install ollama
   3) Guardar la clave en un archivo externo (ver abajo).
 
 -------------------------------------------------------------------------
 CÓMO OCULTAR LA API KEY :
   Crea un archivo llamado  ollama_cloud.key  en la MISMA carpeta que este
   script, y pega dentro SOLO la clave (una línea), sin comillas:
 
        xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
 
 -------------------------------------------------------------------------
=============================================================================
"""

import os
from ollama import Client
 
MODELO = "gpt-oss:120b"                 # Modelo potente en la nube de Ollama.
                                         # Otras opciones: "qwen3-coder:480b-cloud",
                                         # "deepseek-r1:671b", "kimi-k2.6"...
ARCHIVO_CLAVE = "ollama_cloud.key"      # Archivo externo con la API key.
 
 
# ---------------------------------------------------------------------------
# 0) LEER LA API KEY DESDE EL ARCHIVO EXTERNO
# ---------------------------------------------------------------------------
def leer_api_key():
    # Prioridad 1: archivo externo junto al script.
    ruta = os.path.join(os.path.dirname(os.path.abspath(__file__)), ARCHIVO_CLAVE)
    if os.path.exists(ruta):
        with open(ruta, "r", encoding="utf-8") as f:
            clave = f.read().strip()
        if clave:
            return clave
    # Prioridad 2: variable de entorno.
    clave = os.environ.get("OLLAMA_API_KEY")
    if clave:
        return clave
    # Si no hay ninguna, avisamos con claridad.
    raise SystemExit(
        f"[ERROR] No encuentro la API key.\n"
        f"  Crea el archivo '{ARCHIVO_CLAVE}' con tu clave dentro, "
        f"o define la variable de entorno OLLAMA_API_KEY."
    )
 
 
API_KEY = leer_api_key()
 
# El cliente apunta a la nube de Ollama con la clave en la cabecera Authorization.
cliente = Client(
    host="https://ollama.com",
    headers={"Authorization": "Bearer " + API_KEY},
)
 
 
# ---------------------------------------------------------------------------
# 1) LLAMADA COMPLETA CON TODOS LOS PARÁMETROS
# ---------------------------------------------------------------------------
# En la nube, los parámetros de MUESTREO (temperature, top_p...) sí aplican.
# Los de HARDWARE (num_gpu, low_vram, use_mmap, main_gpu, numa...) NO tienen
# sentido aquí, porque el modelo corre en los servidores de Ollama, no en tu
# máquina. Los dejamos comentados para que sepas que existen pero no se usan.
 
def llamada_completa():
    mensajes = [
        {"role": "system", "content": "Eres un profesor de IA claro y conciso."},
        {"role": "user",   "content": "Explica en 3 frases qué es un modelo de lenguaje grande (LLM)."},
    ]
 
    opciones = {
        # ---------------- MUESTREO / CREATIVIDAD ----------------
        "temperature":       0.7,   # Aleatoriedad. Más alto = más creativo. (def: 0.8)
        "top_k":             40,    # Considera solo los K tokens más probables. (def: 40)
        "top_p":             0.9,   # Nucleus sampling. (def: 0.9)
        "min_p":             0.0,   # Prob. mínima relativa al mejor token. (def: 0.0)
        "typical_p":         1.0,   # Muestreo típico. (def: 1.0)
        "seed":              42,    # Semilla para reproducibilidad. (def: 0)
 
        # ---------------- LONGITUD Y REPETICIÓN ----------------
        "num_predict":       300,   # Máx. tokens a generar. -1 = ilimitado. (def: -1)
        "stop":              ["<end>"],  # Secuencias que cortan la generación.
        "repeat_last_n":     64,    # Ventana para penalizar repeticiones. (def: 64)
        "repeat_penalty":    1.1,   # Cuánto penaliza repetir. (def: 1.1)
        "presence_penalty":  0.0,   # Penaliza tokens ya presentes. (def: 0.0)
        "frequency_penalty": 0.0,   # Penaliza por frecuencia. (def: 0.0)
 
        # ---------------- MIROSTAT ----------------
        "mirostat":          0,     # 0 = off, 1 = Mirostat, 2 = Mirostat 2.0. (def: 0)
        "mirostat_tau":      5.0,   # Coherencia vs diversidad. (def: 5.0)
        "mirostat_eta":      0.1,   # Velocidad de adaptación. (def: 0.1)
 
        # ---------------- CONTEXTO ----------------
        "num_ctx":           8192,  # Ventana de contexto. En la nube puedes permitirte más.
 
        # ---------------- PARÁMETROS DE HARDWARE (NO aplican en Cloud) ----------------
        # "num_gpu":   -1,     # (local) capas en GPU
        # "low_vram":  False,  # (local) modo bajo consumo de VRAM
        # "use_mmap":  True,   # (local) mapear modelo en memoria
        # "main_gpu":  0,      # (local) GPU principal
        # "numa":      False,  # (local) optimización NUMA
    }
 
    respuesta = cliente.chat(
        model=MODELO,
        messages=mensajes,
        options=opciones,
        stream=False,
        keep_alive="5m",     # Cuánto mantener el modelo "caliente" tras responder.
        format="",           # "" texto | "json" fuerza JSON | dict = esquema JSON estricto
    )
 
    print("=== RESPUESTA (Cloud) ===")
    print(respuesta.message.content)
 
 
# ---------------------------------------------------------------------------
# 2) STREAMING (recomendado en modelos grandes: la respuesta tarda más)
# ---------------------------------------------------------------------------
def llamada_streaming():
    print("\n=== STREAMING (Cloud) ===")
    flujo = cliente.chat(
        model=MODELO,
        messages=[{"role": "user", "content": "Da 3 aplicaciones reales de los LLM."}],
        options={"temperature": 0.6},
        stream=True,
    )
    for trozo in flujo:
        print(trozo.message.content, end="", flush=True)
    print()
 
 
# ---------------------------------------------------------------------------
# 3) TOOL CALLING NATIVO (esto SÍ funciona con gpt-oss, a diferencia de gemma3)
# ---------------------------------------------------------------------------
def tool_calling_nativo():
    print("\n=== TOOL CALLING NATIVO (Cloud) ===")
 
    # Herramienta real en Python:
    def obtener_clima(ciudad: str) -> str:
        datos = {"Madrid": "28°C soleado", "Bogotá": "18°C nublado"}
        return datos.get(ciudad, "sin datos")
 
    # Definición de la herramienta en formato JSON (esquema de función):
    herramientas = [{
        "type": "function",
        "function": {
            "name": "obtener_clima",
            "description": "Devuelve el clima actual de una ciudad",
            "parameters": {
                "type": "object",
                "properties": {
                    "ciudad": {"type": "string", "description": "Nombre de la ciudad"}
                },
                "required": ["ciudad"],
            },
        },
    }]
 
    mensajes = [{"role": "user", "content": "¿Qué tiempo hace en Madrid?"}]
 
    # 1ª llamada: el modelo decide si usa la herramienta.
    respuesta = cliente.chat(model=MODELO, messages=mensajes, tools=herramientas)
 
    if respuesta.message.tool_calls:
        # El modelo pidió llamar a una función. La ejecutamos nosotros.
        mensajes.append(respuesta.message)  # guardamos su petición en el historial
        for llamada in respuesta.message.tool_calls:
            nombre = llamada.function.name
            args = llamada.function.arguments
            print(f"El modelo pide: {nombre}({args})")
            if nombre == "obtener_clima":
                resultado = obtener_clima(**args)
                mensajes.append({"role": "tool", "content": resultado, "tool_name": nombre})
 
        # 2ª llamada: le damos el resultado y redacta la respuesta final.
        final = cliente.chat(model=MODELO, messages=mensajes)
        print("Respuesta final:", final.message.content)
    else:
        print("El modelo respondió sin usar herramientas:", respuesta.message.content)
 
 
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    llamada_completa()
    llamada_streaming()
    tool_calling_nativo()