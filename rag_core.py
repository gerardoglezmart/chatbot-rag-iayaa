"""Núcleo del chatbot RAG + LLM para el curso IAyAA (MNA, Tec de Monterrey).

Este módulo concentra el pipeline completo extraído del notebook de la
Actividad 5 (Semanas 7 y 8): carga del corpus, limpieza, chunking,
recuperación semántica con sentence-transformers (con TF-IDF como
baseline comparativo) y generación de respuesta con backends
intercambiables:

- ``openai``     -> gpt-5-mini vía API de OpenAI (propietario).
- ``ollama``     -> llama3.2:3b local vía Ollama (código abierto).
- ``hf``         -> Qwen2.5-7B-Instruct vía Hugging Face Inference API (código abierto).
- ``extractivo`` -> respaldo local sin LLM (extrae oraciones del top-1).

El backend se selecciona con la variable de entorno ``LLM_BACKEND``
(``auto`` por defecto: usa el primero disponible en el orden
openai -> hf -> ollama -> extractivo).
"""

from __future__ import annotations

import os
import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------------------------------------------------------------------------
# Configuración general
# ---------------------------------------------------------------------------

_CANDIDATE_DATA_DIRS = [
    Path(__file__).parent / "04_datos_rag",
    Path("04_datos_rag"),
    Path("/content/drive/MyDrive/Colab Notebooks NLP/Actividad 5/04_datos_rag"),
]
DATA_DIR = next((p for p in _CANDIDATE_DATA_DIRS if p.exists()), _CANDIDATE_DATA_DIRS[0])

CHUNK_SIZE = 900
CHUNK_OVERLAP = 150
DEFAULT_TOP_K = 5

EMBEDDING_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"

LLM_BACKEND = os.environ.get("LLM_BACKEND", "auto")
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-5-mini")
OPENAI_REASONING_EFFORT = os.environ.get("OPENAI_REASONING_EFFORT", "low")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.2:3b")
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1")
HF_MODEL = os.environ.get("HF_MODEL", "Qwen/Qwen2.5-7B-Instruct")

TOPIC_MAP = {
    "01_introduccion_ia_teoria": "Introducción a IA y aprendizaje automático",
    "02_gradiente_descendente": "Regresión y gradiente descendente",
    "03_regresion_logistica": "Clasificación y regresión logística",
    "04_entropia_ganancia_informacion": "Árboles de decisión, entropía y ganancia de información",
    "05_sesgo_varianza_curvas_aprendizaje": "Sesgo-varianza y curvas de aprendizaje",
    "06_svm_kernel_holgura": "SVM, kernel radial y variables de holgura",
    "07_introduccion_ia_teoria": "Introducción a IA y aprendizaje automático",
    "08_metricas_clasificacion_matriz_confusion": "Métricas de clasificación y matriz de confusión",
}

PROMPT_BASE = """Rol: Eres un asistente académico del curso Inteligencia Artificial y Aprendizaje Automático (IAyAA).

Objetivo:
- Responder preguntas del curso apoyándote en el contexto recuperado del material oficial.

Instrucciones obligatorias:
- Responde siempre en español.
- Basa tu respuesta en la información del contexto recuperado; puedes reorganizarla y sintetizarla con tus propias palabras.
- Si el contexto cubre el tema solo de forma parcial, responde con lo que sí está presente y aclara en una frase qué aspecto no quedó cubierto.
- Únicamente si el contexto no contiene nada relevante para la pregunta, inicia con: 'Con el contexto recuperado no es posible responder con suficiente precisión'.
- No inventes definiciones, fórmulas o ejemplos ajenos al contexto.
- Tono claro y preciso para un estudiante de maestría; menciona de forma breve la fuente o tema del que sale la respuesta.

Formato de salida esperado:
- Respuesta de 3 a 6 oraciones.
"""

# ---------------------------------------------------------------------------
# Carga y preparación del corpus
# ---------------------------------------------------------------------------


def infer_topic(path: Path) -> str:
    return TOPIC_MAP.get(path.stem, "Tema por clasificar")


def read_txt(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def read_pdf(path: Path) -> str:
    reader = PdfReader(str(path))
    pages = [(page.extract_text() or "") for page in reader.pages]
    return "\n".join(pages)


def load_document(path: Path) -> dict:
    suffix = path.suffix.lower()
    if suffix == ".txt":
        text = read_txt(path)
    elif suffix == ".pdf":
        text = read_pdf(path)
    else:
        raise ValueError(f"Tipo de archivo no soportado: {path.name}")

    return {
        "source": path.name,
        "text": text,
        "tipo": suffix.lstrip("."),
        "tema": infer_topic(path),
    }


def load_corpus(data_dir: Path = DATA_DIR, allowed_suffixes=(".txt", ".pdf")) -> pd.DataFrame:
    documents = []
    for path in sorted(data_dir.iterdir()):
        if path.is_file() and path.suffix.lower() in allowed_suffixes:
            documents.append(load_document(path))
    return pd.DataFrame(documents)


def clean_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[\t\f\v]+", " ", text)
    text = re.sub(r" ?\n ?", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r" {2,}", " ", text)
    return text.strip()


def prepare_corpus(documents: pd.DataFrame, min_chars: int = 80) -> pd.DataFrame:
    prepared = documents.copy()
    prepared["text"] = prepared["text"].fillna("").map(clean_text)
    prepared["num_chars_clean"] = prepared["text"].str.len()
    prepared = prepared[prepared["num_chars_clean"] >= min_chars].reset_index(drop=True)
    return prepared


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------


def split_into_paragraphs(text: str) -> list[str]:
    paragraphs = [paragraph.strip() for paragraph in re.split(r"\n{2,}", text) if paragraph.strip()]
    return paragraphs or ([text.strip()] if text.strip() else [])


def split_long_paragraph(paragraph: str, chunk_size: int) -> list[str]:
    words = paragraph.split()
    windows = []
    current_words = []

    for word in words:
        candidate = " ".join(current_words + [word]).strip()
        if current_words and len(candidate) > chunk_size:
            windows.append(" ".join(current_words).strip())
            current_words = [word]
        else:
            current_words.append(word)

    if current_words:
        windows.append(" ".join(current_words).strip())

    return [window for window in windows if window]


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP) -> list[str]:
    paragraphs = split_into_paragraphs(text)
    chunks = []
    current_parts = []

    for paragraph in paragraphs:
        if len(paragraph) > chunk_size:
            if current_parts:
                chunks.append("\n\n".join(current_parts).strip())
                current_parts = []
            chunks.extend(split_long_paragraph(paragraph, chunk_size))
            continue

        candidate_parts = current_parts + [paragraph]
        candidate_text = "\n\n".join(candidate_parts).strip()

        if current_parts and len(candidate_text) > chunk_size:
            chunks.append("\n\n".join(current_parts).strip())
            if chunk_overlap > 0:
                overlap_text = chunks[-1][-chunk_overlap:].strip()
                current_parts = [overlap_text, paragraph] if overlap_text else [paragraph]
            else:
                current_parts = [paragraph]
        else:
            current_parts = candidate_parts

    if current_parts:
        chunks.append("\n\n".join(current_parts).strip())

    return [chunk for chunk in chunks if chunk]


def build_chunks(corpus: pd.DataFrame, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP) -> pd.DataFrame:
    rows = []
    for _, row in corpus.iterrows():
        text_chunks = chunk_text(row["text"], chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        for chunk_index, chunk in enumerate(text_chunks, start=1):
            rows.append(
                {
                    "chunk_id": f"{Path(row['source']).stem}_chunk_{chunk_index:03d}",
                    "source": row["source"],
                    "tipo": row["tipo"],
                    "tema": row["tema"],
                    "chunk_index": chunk_index,
                    "text": chunk,
                    "num_chars_chunk": len(chunk),
                    "num_words_chunk": len(chunk.split()),
                }
            )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Índice semántico (sentence-transformers) + baseline TF-IDF
# ---------------------------------------------------------------------------

_INDEX: dict = {}


def build_index(data_dir: Path = DATA_DIR, with_tfidf: bool = True) -> dict:
    """Construye (una sola vez) el índice de recuperación sobre el corpus."""
    if _INDEX.get("ready"):
        return _INDEX

    from sentence_transformers import SentenceTransformer

    documents_df = load_corpus(data_dir)
    prepared_df = prepare_corpus(documents_df)
    chunks_df = build_chunks(prepared_df)

    st_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    semantic_embeddings = st_model.encode(
        chunks_df["text"].tolist(),
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    _INDEX.update(
        {
            "ready": True,
            "documents_df": documents_df,
            "chunks_df": chunks_df,
            "st_model": st_model,
            "semantic_embeddings": semantic_embeddings,
        }
    )

    if with_tfidf:
        vectorizer = TfidfVectorizer(
            lowercase=True,
            strip_accents="unicode",
            ngram_range=(1, 2),
            min_df=1,
            max_df=0.95,
            token_pattern=r"(?u)\b[a-záéíóúüñ]{2,}\b",
            sublinear_tf=True,
        )
        _INDEX["tfidf_vectorizer"] = vectorizer
        _INDEX["tfidf_embeddings"] = vectorizer.fit_transform(chunks_df["text"])

    return _INDEX


def recuperar_contexto(pregunta: str, k: int = DEFAULT_TOP_K) -> pd.DataFrame:
    """Recuperación semántica principal: similitud coseno sobre embeddings."""
    index = build_index()
    query_emb = index["st_model"].encode([pregunta], normalize_embeddings=True)
    similarities = (index["semantic_embeddings"] @ query_emb.T).flatten()
    top_indices = similarities.argsort()[::-1][:k]

    results = index["chunks_df"].iloc[top_indices].copy()
    results["score_similitud"] = similarities[top_indices]
    results["pregunta"] = pregunta
    return results[
        ["pregunta", "chunk_id", "source", "tema", "chunk_index", "score_similitud", "text"]
    ].reset_index(drop=True)


def recuperar_contexto_tfidf(pregunta: str, k: int = DEFAULT_TOP_K) -> pd.DataFrame:
    """Baseline léxico (TF-IDF) conservado para comparación en el informe."""
    index = build_index()
    query_vec = index["tfidf_vectorizer"].transform([pregunta])
    similarities = cosine_similarity(query_vec, index["tfidf_embeddings"]).flatten()
    top_indices = similarities.argsort()[::-1][:k]

    results = index["chunks_df"].iloc[top_indices].copy()
    results["score_similitud"] = similarities[top_indices]
    results["pregunta"] = pregunta
    return results[
        ["pregunta", "chunk_id", "source", "tema", "chunk_index", "score_similitud", "text"]
    ].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------


def format_contexto(contexto_df: pd.DataFrame) -> str:
    bloques = []
    for row in contexto_df.itertuples(index=False):
        bloques.append(
            f"[Fuente: {row.source} | Chunk: {row.chunk_id} | Score: {row.score_similitud:.4f}]\n{row.text}"
        )
    return "\n\n".join(bloques)


def construir_prompt(pregunta: str, contexto_df: pd.DataFrame) -> str:
    contexto = format_contexto(contexto_df)
    return (
        f"{PROMPT_BASE}\n"
        "Contexto recuperado:\n"
        "--------------------\n"
        f"{contexto}\n\n"
        "Pregunta del usuario:\n"
        "---------------------\n"
        f"{pregunta}\n\n"
        "Respuesta final:\n"
    )


# ---------------------------------------------------------------------------
# Backends de generación
# ---------------------------------------------------------------------------


def get_openai_llm_callable():
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return None

    from openai import OpenAI

    client = OpenAI(api_key=api_key)

    def llm_openai(prompt: str) -> str:
        response = client.responses.create(
            model=OPENAI_MODEL,
            input=prompt,
            reasoning={"effort": OPENAI_REASONING_EFFORT},
            store=False,
        )
        return response.output_text.strip()

    return llm_openai


def get_ollama_llm_callable():
    """Modelo open-source local servido por Ollama (API compatible con OpenAI)."""
    import urllib.request

    try:
        urllib.request.urlopen(OLLAMA_BASE_URL.removesuffix("/v1"), timeout=2)
    except Exception:
        return None

    from openai import OpenAI

    client = OpenAI(base_url=OLLAMA_BASE_URL, api_key="ollama")

    def llm_ollama(prompt: str) -> str:
        response = client.chat.completions.create(
            model=OLLAMA_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        )
        return response.choices[0].message.content.strip()

    return llm_ollama


def get_hf_llm_callable():
    """Modelo open-source en la nube vía Hugging Face Inference API."""
    token = os.environ.get("HF_TOKEN")
    if not token:
        return None

    from huggingface_hub import InferenceClient

    client = InferenceClient(model=HF_MODEL, token=token)

    def llm_hf(prompt: str) -> str:
        response = client.chat_completion(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=500,
            temperature=0.2,
        )
        return response.choices[0].message.content.strip()

    return llm_hf


_BACKEND_FACTORIES = {
    "openai": get_openai_llm_callable,
    "hf": get_hf_llm_callable,
    "ollama": get_ollama_llm_callable,
}


def get_llm_callable(backend: str = LLM_BACKEND) -> tuple:
    """Devuelve ``(callable, nombre_backend)``; ``(None, 'extractivo')`` si no hay LLM."""
    backend = (backend or "auto").lower()
    if backend == "extractivo":
        return None, "extractivo"

    if backend in _BACKEND_FACTORIES:
        return _BACKEND_FACTORIES[backend](), backend

    for name in ("openai", "hf", "ollama"):  # orden de preferencia en modo auto
        llm = _BACKEND_FACTORIES[name]()
        if llm is not None:
            return llm, name
    return None, "extractivo"


def extractive_fallback(contexto_df: pd.DataFrame, max_sentences: int = 2) -> str:
    if contexto_df.empty:
        return "No encontré contexto suficiente en el corpus para responder con confianza."

    top_text = contexto_df.iloc[0]["text"]
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", top_text) if s.strip()]
    selected = sentences[:max_sentences] if sentences else [top_text[:320].strip()]
    return f"Respuesta construida directamente a partir del fragmento recuperado: {' '.join(selected).strip()}"


# ---------------------------------------------------------------------------
# Orquestación
# ---------------------------------------------------------------------------


def generar_respuesta(pregunta: str, contexto_df: pd.DataFrame, llm_callable=None) -> dict:
    prompt = construir_prompt(pregunta, contexto_df)

    if llm_callable is None:
        return {
            "respuesta": extractive_fallback(contexto_df),
            "prompt": prompt,
            "modo_generacion": "extractivo_local",
            "error_generacion": None,
        }

    try:
        return {
            "respuesta": llm_callable(prompt),
            "prompt": prompt,
            "modo_generacion": "llm",
            "error_generacion": None,
        }
    except Exception as exc:
        return {
            "respuesta": extractive_fallback(contexto_df),
            "prompt": prompt,
            "modo_generacion": "respaldo_por_error",
            "error_generacion": f"{type(exc).__name__}: {exc}",
        }


def chatbot_rag(pregunta: str, k: int = DEFAULT_TOP_K, backend: str = LLM_BACKEND) -> dict:
    contexto_df = recuperar_contexto(pregunta, k=k)
    llm_callable, backend_usado = get_llm_callable(backend)
    generacion = generar_respuesta(pregunta, contexto_df, llm_callable=llm_callable)

    return {
        "pregunta": pregunta,
        "respuesta": generacion["respuesta"],
        "prompt": generacion["prompt"],
        "modo_generacion": backend_usado if generacion["modo_generacion"] == "llm" else generacion["modo_generacion"],
        "error_generacion": generacion["error_generacion"],
        "contexto": contexto_df,
        "fuentes": contexto_df[["chunk_id", "source", "tema", "score_similitud"]].copy(),
    }


# ---------------------------------------------------------------------------
# Batería oficial de preguntas (sección 7 del notebook)
# ---------------------------------------------------------------------------

BATERIA_PREGUNTAS = [
    {
        "id": "P1",
        "tipo": "fácil",
        "tema": "gradiente descendente",
        "pregunta": "¿Qué es el gradiente descendente y para qué sirve en aprendizaje automático?",
        "fuente_esperada": "02_gradiente_descendente.txt",
    },
    {
        "id": "P2",
        "tipo": "promedio",
        "tema": "métricas de clasificación",
        "pregunta": "¿Qué información aporta una matriz de confusión al evaluar un clasificador?",
        "fuente_esperada": "08_metricas_clasificacion_matriz_confusion.pdf",
    },
    {
        "id": "P3",
        "tipo": "promedio",
        "tema": "sesgo-varianza",
        "pregunta": "¿Qué describe el dilema sesgo-varianza y por qué es importante al entrenar modelos?",
        "fuente_esperada": "05_sesgo_varianza_curvas_aprendizaje.txt",
    },
    {
        "id": "P4",
        "tipo": "difícil (comparativa)",
        "tema": "árboles de decisión",
        "pregunta": "¿En qué se diferencian la entropía y la ganancia de información dentro de un árbol de decisión?",
        "fuente_esperada": "04_entropia_ganancia_informacion.txt",
    },
    {
        "id": "P5",
        "tipo": "difícil (comparativa)",
        "tema": "clasificación y márgenes",
        "pregunta": "¿Qué diferencia práctica hay entre usar regresión logística y una SVM para clasificación?",
        "fuente_esperada": "03_regresion_logistica.txt + 06_svm_kernel_holgura.txt",
    },
    {
        "id": "P6",
        "tipo": "difícil (integradora)",
        "tema": "SVM y generalización",
        "pregunta": "Si un modelo no separa perfectamente los datos, ¿qué papel cumplen las variables de holgura y el kernel radial en una SVM?",
        "fuente_esperada": "06_svm_kernel_holgura.txt",
    },
]


def evaluar_hit_rate(bateria: list[dict] = BATERIA_PREGUNTAS, k: int = DEFAULT_TOP_K, retriever=recuperar_contexto) -> pd.DataFrame:
    """Evaluación cuantitativa: ¿la fuente esperada aparece en el top-k / top-1?"""
    filas = []
    for item in bateria:
        contexto = retriever(item["pregunta"], k=k)
        fuentes_recuperadas = set(contexto["source"])
        esperadas = {f.strip() for f in item["fuente_esperada"].split("+")}
        filas.append(
            {
                "id": item["id"],
                "tipo": item["tipo"],
                "hit_at_k": esperadas.issubset(fuentes_recuperadas),
                "hit_at_1": contexto.iloc[0]["source"] in esperadas,
                "score_top1": round(float(contexto.iloc[0]["score_similitud"]), 4),
                "top1_source": contexto.iloc[0]["source"],
            }
        )
    return pd.DataFrame(filas)
