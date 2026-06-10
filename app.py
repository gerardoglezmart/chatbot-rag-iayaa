"""Interfaz Gradio del chatbot RAG + LLM del curso IAyAA.

Corre igual en local (``python app.py``) y en Hugging Face Spaces.
El backend de generación se elige en la interfaz o con la variable
de entorno ``LLM_BACKEND`` (openai | ollama | hf | extractivo | auto).
"""

import os

import gradio as gr

import rag_core

TOP_K = int(os.environ.get("RAG_TOP_K", rag_core.DEFAULT_TOP_K))

BACKEND_CHOICES = {
    "Automático (primero disponible)": "auto",
    "OpenAI gpt-5-mini (propietario)": "openai",
    "Open-source en la nube — Qwen2.5-7B vía HF": "hf",
    "Open-source local — llama3.2:3b vía Ollama": "ollama",
    "Extractivo local (sin LLM)": "extractivo",
}

EJEMPLOS = [item["pregunta"] for item in rag_core.BATERIA_PREGUNTAS]

print("Construyendo índice semántico del corpus...")
rag_core.build_index()
print("Índice listo.")


def responder_chat(mensaje: str, historial, backend_label: str) -> str:
    backend = BACKEND_CHOICES.get(backend_label, "auto")
    resultado = rag_core.chatbot_rag(mensaje, k=TOP_K, backend=backend)

    fuentes = resultado["fuentes"].drop_duplicates("source")
    lista_fuentes = "\n".join(
        f"- `{r.source}` — {r.tema} (score {r.score_similitud:.3f})"
        for r in fuentes.itertuples(index=False)
    )

    pie = f"\n\n**📚 Fuentes consultadas:**\n{lista_fuentes}\n\n*Modo de generación: `{resultado['modo_generacion']}`*"
    if resultado["error_generacion"]:
        pie += f"\n*Aviso: el LLM falló y se usó el respaldo extractivo ({resultado['error_generacion']}).*"
    return resultado["respuesta"] + pie


demo = gr.ChatInterface(
    fn=responder_chat,
    title="🤖 Chatbot académico IAyAA — RAG + LLM",
    description=(
        "Pregunta sobre los temas del curso *Inteligencia Artificial y Aprendizaje Automático*: "
        "gradiente descendente, regresión logística, entropía y ganancia de información, "
        "sesgo-varianza, SVM y métricas de clasificación. Las respuestas se generan con base "
        "en el material del curso y citan sus fuentes.\n\n"
        "Maestría en Inteligencia Artificial Aplicada — Tec de Monterrey — Actividad 5 (Semanas 7 y 8)."
    ),
    examples=EJEMPLOS,
    additional_inputs=[
        gr.Dropdown(
            choices=list(BACKEND_CHOICES.keys()),
            value="Automático (primero disponible)",
            label="Modelo generativo",
        )
    ],
)

if __name__ == "__main__":
    demo.launch()
