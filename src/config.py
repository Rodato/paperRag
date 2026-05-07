import logging
import os
from pathlib import Path
from typing import List, Optional

from dotenv import load_dotenv
from langchain_core.embeddings import Embeddings
from langchain_openai import ChatOpenAI
from openai import OpenAI

logger = logging.getLogger(__name__)

load_dotenv()

OPENROUTER_BASE = "https://openrouter.ai/api/v1"
OPENROUTER_KEY = os.getenv("OPENROUTER_API_KEY")

DATA_DIR = Path(__file__).parent.parent / "data" / "papers"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Modelos
PROCESSING_MODEL = "google/gemini-2.5-flash-lite"
EMBEDDING_MODEL = "openai/text-embedding-ada-002"

QUERY_MODELS = {
    "GPT-4o Mini": "openai/gpt-4o-mini",
    "Gemini 2.5 Flash Lite": "google/gemini-2.5-flash-lite",
    "Llama 3.3 70B": "meta-llama/llama-3.3-70b-instruct",
    "DeepSeek R1 Qwen 32B": "deepseek/deepseek-r1-distill-qwen-32b",
    "Ministral 14B": "mistralai/ministral-14b-2512",
}


class OpenRouterEmbeddings(Embeddings):
    """Embeddings via OpenRouter usando el cliente openai directamente."""

    def __init__(self, model: str, api_key: str, base_url: str):
        self._client = OpenAI(api_key=api_key, base_url=base_url)
        self._model = model

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        response = self._client.embeddings.create(
            input=texts, model=self._model, encoding_format="float"
        )
        return [item.embedding for item in response.data]

    def embed_query(self, text: str) -> List[float]:
        response = self._client.embeddings.create(
            input=[text], model=self._model, encoding_format="float"
        )
        return response.data[0].embedding


def get_processing_llm() -> ChatOpenAI:
    """LLM para el pipeline de procesamiento de PDFs (extracción de título, secciones, referencias)."""
    return ChatOpenAI(
        model=PROCESSING_MODEL,
        base_url=OPENROUTER_BASE,
        api_key=OPENROUTER_KEY,
        temperature=0,
    )


def get_query_llm(model_id: Optional[str] = None) -> ChatOpenAI:
    """LLM para el agente RAG (generación de respuestas)."""
    model = model_id or next(iter(QUERY_MODELS.values()))
    return ChatOpenAI(
        model=model,
        base_url=OPENROUTER_BASE,
        api_key=OPENROUTER_KEY,
        temperature=0,
    )


def get_embeddings() -> OpenRouterEmbeddings:
    """Embeddings via OpenRouter (openai/text-embedding-ada-002)."""
    return OpenRouterEmbeddings(
        model=EMBEDDING_MODEL,
        api_key=OPENROUTER_KEY,
        base_url=OPENROUTER_BASE,
    )
