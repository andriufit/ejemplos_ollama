#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=============================================================================
 PRUEBA DE GEMMA 3 4B EN OLLAMA (LOCAL)  -  Material didáctico
=============================================================================
 Objetivo: mostrar TODOS los parámetros posibles de una llamada al modelo,
 explicados uno a uno, para la asignatura de Modelos de IA.

 Requisitos previos:
   1) Tener Ollama instalado y corriendo (ollama.com/download)
   2) Haber descargado el modelo:   ollama run gemma3:4b   (o: ollama pull gemma3:4b)
   3) Instalar la librería de Python:   pip install ollama

 Cómo ejecutar:   python prueba_gemma3_ollama.py

 NOTA DE HARDWARE (portátil con RTX 4050 6GB VRAM + 8GB RAM):
   - Un modelo 4B en Q4 cabe en la GPU. Mantén num_ctx moderado (2048-4096).
   - Subir num_ctx a 32768 dispara el consumo de RAM/VRAM y puede tirar a CPU.
=============================================================================
"""

import ollama

MODELO = "gemma3:4b"


# ---------------------------------------------------------------------------
# 1) LLAMADA COMPLETA CON TODOS LOS PARÁMETROS
# ---------------------------------------------------------------------------
# La función chat() acepta parámetros de ALTO NIVEL (model, messages, stream,
# format, keep_alive...) y un diccionario "options" con TODOS los parámetros
# de muestreo (sampling) y de ejecución (runtime) del motor.
# Aquí los ponemos TODOS con un comentario y su valor por defecto.

def llamada_completa():
    # --- Conversación (lista de mensajes con roles: system / user / assistant) ---
    mensajes = [
        {"role": "system",  "content": "Eres un profesor de IA que explica con ejemplos sencillos."},
        {"role": "user",    "content": "Explica en 2 frases qué es el sobreajuste (overfitting)."},
    ]

    # --- Diccionario de OPTIONS: parámetros de muestreo y de motor ---
    opciones = {
        # ---------------- MUESTREO / CREATIVIDAD ----------------
        "temperature":       0.8,   # Aleatoriedad. Más alto = más creativo. (def: 0.8)
        "top_k":             40,    # Considera solo los K tokens más probables. (def: 40)
        "top_p":             0.9,   # Nucleus sampling: masa de probabilidad acumulada. (def: 0.9)
        "min_p":             0.0,   # Prob. mínima relativa al token más probable. (def: 0.0)
        "typical_p":         1.0,   # Muestreo "típico" (locally typical). (def: 1.0)
        "tfs_z":             1.0,   # Tail-Free Sampling. 1.0 = desactivado. (OBSOLETO en versiones nuevas)
        "seed":              42,    # Semilla. Fija un valor para resultados reproducibles. (def: 0)

        # ---------------- LONGITUD Y REPETICIÓN ----------------
        "num_predict":       256,   # Máx. tokens a generar. -1 = ilimitado, -2 = llenar contexto. (def: -1)
        "stop":              ["user:", "<end>"],  # Secuencias que detienen la generación.
        "repeat_last_n":     64,    # Cuántos tokens atrás mira para penalizar repeticiones. (def: 64)
        "repeat_penalty":    1.1,   # Cuánto penaliza repetir. >1 penaliza más. (def: 1.1)
        "presence_penalty":  0.0,   # Penaliza tokens ya usados (estilo OpenAI). (def: 0.0)
        "frequency_penalty": 0.0,   # Penaliza según frecuencia de aparición. (def: 0.0)
        "penalize_newline":  True,  # Penalizar saltos de línea. (def: True)

        # ---------------- MIROSTAT (control de perplejidad) ----------------
        "mirostat":          0,     # 0 = off, 1 = Mirostat, 2 = Mirostat 2.0. (def: 0)
        "mirostat_tau":      5.0,   # Equilibrio coherencia/diversidad. Menor = más enfocado. (def: 5.0)
        "mirostat_eta":      0.1,   # Velocidad de adaptación del algoritmo. (def: 0.1)

        # ---------------- CONTEXTO Y RENDIMIENTO ----------------
        "num_ctx":           4096,  # Tamaño de la ventana de contexto. (def: 2048/4096 según versión)
        "num_batch":         512,   # Tamaño de lote de procesamiento del prompt. (def: 512)
        "num_keep":          4,     # Tokens iniciales del prompt que se conservan siempre.
        "num_gpu":           -1,    # Nº de capas a descargar en GPU. -1 = automático (recomendado).
        "main_gpu":          0,     # GPU principal si hay varias. (def: 0)
        "low_vram":          False, # Modo bajo consumo de VRAM. Actívalo si te quedas sin memoria.
        "num_thread":        0,     # Hilos de CPU. 0 = automático. (def: 0)
        "numa":              False, # Optimización NUMA (servidores multi-socket). (def: False)

        # ---------------- MEMORIA / CARGA ----------------
        "use_mmap":          True,  # Mapear el modelo en memoria (carga más rápida). (def: True)
        "use_mlock":         False, # Bloquear el modelo en RAM (evita swap). (def: False)
        "vocab_only":        False, # Cargar solo el vocabulario, sin pesos (raro). (def: False)
    }

    respuesta = ollama.chat(
        model=MODELO,
        messages=mensajes,
        options=opciones,        # <-- todos los parámetros de arriba
        stream=False,            # False = respuesta completa de golpe; True = por trozos (ver más abajo)
        keep_alive="5m",         # Cuánto mantener el modelo en memoria tras responder.
                                 #   "5m" = 5 min | 0 = descargar ya | -1 = mantener siempre
        format="",               # "" = texto libre | "json" = fuerza JSON | dict = esquema JSON estricto
        # think=False,           # Solo modelos "de razonamiento". Gemma 3 NO lo es -> déjalo fuera.
        # tools=[...],           # ¡OJO! gemma3:4b NO admite tools nativas -> daría error. Ver sección 4.
    )

    print("=== RESPUESTA ===")
    print(respuesta["message"]["content"])

    # La respuesta trae también métricas útiles para enseñar rendimiento:
    print("\n=== MÉTRICAS ===")
    print(f"Tokens generados : {respuesta.get('eval_count')}")
    print(f"Duración total   : {respuesta.get('total_duration', 0) / 1e9:.2f} s")


# ---------------------------------------------------------------------------
# 2) VERSIÓN EN STREAMING (respuesta token a token, como un chat en vivo)
# ---------------------------------------------------------------------------
def llamada_streaming():
    print("\n=== STREAMING ===")
    flujo = ollama.chat(
        model=MODELO,
        messages=[{"role": "user", "content": "Enumera 3 tipos de aprendizaje automático."}],
        options={"temperature": 0.7, "num_ctx": 2048},
        stream=True,   # <-- clave: activa la respuesta progresiva
    )
    for trozo in flujo:
        print(trozo["message"]["content"], end="", flush=True)
    print()


# ---------------------------------------------------------------------------
# 3) SALIDA ESTRUCTURADA EN JSON (útil para pipelines / demos en clase)
# ---------------------------------------------------------------------------
def salida_json():
    print("\n=== SALIDA JSON ===")
    respuesta = ollama.chat(
        model=MODELO,
        messages=[{"role": "user",
                   "content": "Devuelve un JSON con las claves 'concepto' y 'definicion' sobre 'dataset'."}],
        format="json",              # fuerza que la salida sea JSON válido
        options={"temperature": 0}, # temperatura 0 = respuesta determinista, ideal para datos
    )
    print(respuesta["message"]["content"])


# ---------------------------------------------------------------------------
# 4) FUNCTION CALLING CON GEMMA 3 (por prompt, NO con tools= nativas)
# ---------------------------------------------------------------------------
# Como gemma3:4b no soporta el parámetro tools=, el "uso de herramientas" se
# hace pidiéndole en el prompt que devuelva la llamada a la función, y luego
# TÚ la ejecutas en Python. Esto es la esencia del function calling manual.

def herramienta_por_prompt():
    print("\n=== FUNCTION CALLING (por prompt) ===")

    # Nuestra "herramienta" real en Python:
    def obtener_clima(ciudad):
        datos = {"Madrid": "28°C soleado", "Bogotá": "18°C nublado"}
        return datos.get(ciudad, "sin datos")

    prompt = (
        "Tienes una función: obtener_clima(ciudad). la ciudad que le pasas a obtener clima tiene que ir con la primera en mayuscula y el resto en minuscula con sus tildes, si el usuario escribe mal el nombre corrigelo antes de pasarlo a la funcion "
        "Si el usuario pregunta por el clima, responde SOLO con una línea así CON EL NOMBRE CORREGIDO A UN NOMBRE DE UNA CIUDAD REAL CON LA PRIMERA LETRA MAYUSCULA Y EL RESTO EN MINUSCULA RESPETANDO SUS TILDES: "
        "LLAMADA: obtener_clima(\"NOMBRE_CIUDAD\")\n\n"
        "Usuario: ¿Qué tiempo hace en botota?"
    )

    respuesta = ollama.chat(
        model=MODELO,
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": 0},  # determinista para que respete el formato
    )
    texto = respuesta["message"]["content"].strip()
    print("El modelo pidió:", texto)

    # Parseamos y ejecutamos la función si el modelo la solicitó:
    if "obtener_clima" in texto and '"' in texto:
        ciudad = texto.split('"')[1]
        resultado = obtener_clima(ciudad)
        print(f"Resultado real de la función: {resultado}")


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    llamada_completa()
    llamada_streaming()
    salida_json()
    herramienta_por_prompt()
