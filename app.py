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

# Con additional_inputs, cada ejemplo debe ser [mensaje, dropdown, api_key]
EJEMPLOS = [
    [item["pregunta"], "Automático (primero disponible)", ""]
    for item in rag_core.BATERIA_PREGUNTAS
]

print("Construyendo índice semántico del corpus...")
rag_core.build_index()
print("Índice listo.")


def _turnos_usuario(historial) -> list[str]:
    """Extrae los mensajes previos del usuario (soporta formato dict y tuplas)."""
    turnos = []
    for m in historial or []:
        if isinstance(m, dict):
            if m.get("role") == "user" and isinstance(m.get("content"), str):
                turnos.append(m["content"])
        elif isinstance(m, (list, tuple)) and m and isinstance(m[0], str):
            turnos.append(m[0])
    return turnos


def _openai_callable_con_clave(api_key: str):
    """Backend OpenAI efímero con la clave que el usuario pegó en la interfaz.

    La clave solo vive en memoria durante esta petición: no se guarda,
    no se escribe a disco y no aparece en logs.
    """
    from openai import OpenAI

    client = OpenAI(api_key=api_key)

    def llm_openai(prompt: str) -> str:
        response = client.responses.create(
            model=rag_core.OPENAI_MODEL,
            input=prompt,
            reasoning={"effort": rag_core.OPENAI_REASONING_EFFORT},
            store=False,
        )
        return response.output_text.strip()

    return llm_openai


def responder_chat(mensaje: str, historial, backend_label: str, api_key_usuario: str = "") -> str:
    backend = BACKEND_CHOICES.get(backend_label, "auto")

    # Memoria conversacional ligera: las preguntas cortas suelen ser seguimientos
    # ("dame una definición más corta"), así que el retrieval se enriquece con la
    # pregunta anterior y el LLM recibe el hilo para responder en contexto.
    previas = _turnos_usuario(historial)
    consulta_retrieval = mensaje
    pregunta_llm = mensaje
    if previas and len(mensaje.split()) < 12:
        consulta_retrieval = f"{previas[-1]} {mensaje}"
        pregunta_llm = (
            f"{mensaje}\n(Nota: es una pregunta de seguimiento; la pregunta anterior "
            f"del usuario fue: '{previas[-1]}'. Responde la nueva petición en ese contexto.)"
        )

    contexto_df = rag_core.recuperar_contexto(consulta_retrieval, k=TOP_K)

    api_key_usuario = (api_key_usuario or "").strip()
    if api_key_usuario:
        llm_callable, backend_usado = _openai_callable_con_clave(api_key_usuario), "openai (clave del usuario)"
    else:
        llm_callable, backend_usado = rag_core.get_llm_callable(backend)
    generacion = rag_core.generar_respuesta(pregunta_llm, contexto_df, llm_callable=llm_callable)
    resultado = {
        "respuesta": generacion["respuesta"],
        "modo_generacion": backend_usado if generacion["modo_generacion"] == "llm" else generacion["modo_generacion"],
        "error_generacion": generacion["error_generacion"],
        "fuentes": contexto_df[["chunk_id", "source", "tema", "score_similitud"]].copy(),
    }

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
        ),
        gr.Textbox(
            value="",
            type="password",
            label="OPENAI_API_KEY (opcional)",
            placeholder="Pega tu clave para responder con gpt-5-mini; vacío usa el modelo por defecto",
        ),
    ],
)

if __name__ == "__main__":
    demo.launch()
