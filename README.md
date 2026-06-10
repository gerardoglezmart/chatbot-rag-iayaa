---
title: Chatbot RAG IAyAA
emoji: 🤖
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: "6.17.3"
app_file: app.py
pinned: false
---

# Chatbot académico IAyAA — RAG + LLM

Chatbot de pregunta-respuesta sobre el material del curso **Inteligencia Artificial
y Aprendizaje Automático (IAyAA)**, construido con una arquitectura
**RAG (Retrieval-Augmented Generation) + LLM**.

Actividad 5 (Semanas 7 y 8) · Procesamiento de Lenguaje Natural ·
Maestría en Inteligencia Artificial Aplicada · Tecnológico de Monterrey.

## Equipo 3

| Integrante | Matrícula |
|---|---|
| José Emiliano Riosmena Castañón | A01704245 |
| Sofía Donlucas Bañuelos | A01655565 |
| Sebastián Estrada García | A00805402 |
| Gerardo González Martínez | A01840096 |

🚀 **Demo en línea:** https://huggingface.co/spaces/GerardoTec/chatbot-rag-iayaa

## Estructura del repositorio

| Archivo | Descripción |
|---|---|
| `Equipo_03_semanas_7_y_8.ipynb` | **Entregable**: notebook con problema, justificación, implementación, evaluación y conclusiones |
| `rag_core.py` | Pipeline RAG compartido: carga del corpus, chunking, recuperación semántica y backends LLM |
| `app.py` | Interfaz de chat (Gradio), desplegada en Hugging Face Spaces |
| `04_datos_rag/` | Corpus documental del curso (6 `.txt` + 2 `.pdf`) |
| `requirements.txt` | Dependencias del proyecto |

## Arquitectura
<img width="2318" height="3844" alt="image" src="https://github.com/user-attachments/assets/47d6a826-7166-43ac-b27c-5938f61e168c" />

Backends de generación intercambiables (variable `LLM_BACKEND` o selector en la interfaz):

| Backend | Modelo | Tipo |
|---|---|---|
| `openai` | `gpt-5-mini` | Propietario (API de OpenAI) |
| `claude` | `claude-opus-4-8` | Propietario (API de Anthropic) |
| `hf` | `Qwen/Qwen2.5-7B-Instruct` | **Código abierto** (HF Inference API) |
| `ollama` | `llama3.2:3b` | **Código abierto** (local, CPU) |
| `extractivo` | — | Respaldo local sin LLM |

## Uso local

```bash
conda create -n rag-chatbot python=3.11 -y
conda activate rag-chatbot
pip install -r requirements.txt

# Opción A: modelo open-source local
ollama pull llama3.2:3b          # requiere https://ollama.com
export LLM_BACKEND=ollama

# Opción B: OpenAI
export OPENAI_API_KEY=sk-...     # y LLM_BACKEND=openai

python app.py                    # abre http://localhost:7860
```

## Despliegue

Cada `push` a `main` dispara un GitHub Action que sincroniza el repositorio con el
Space de Hugging Face, donde la app se reconstruye automáticamente. Los secretos
(`HF_TOKEN`, `OPENAI_API_KEY`) se configuran en los ajustes del Space, nunca en el código.
