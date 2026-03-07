"""HuggingFace embedding provider implementation."""

import logging


from langchain_core.embeddings import Embeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from src.embeddings.base import EmbeddingProvider

logger = logging.getLogger(__name__)


class HuggingFaceEmbeddingProvider(EmbeddingProvider):
    """Embedding provider using HuggingFace models."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the HuggingFace embedding provider.

        Args:
            model_name: HuggingFace model name.
        """
        self._model_name = model_name
        self._embeddings = HuggingFaceEmbeddings(model_name=model_name)
        logger.info(f"Initialized HuggingFace embeddings with model: {model_name}")

    def get_embeddings(self) -> Embeddings:
        """Get the HuggingFace embeddings instance."""
        return self._embeddings

    @property
    def model_name(self) -> str:
        """Get the embedding model name."""
        return self._model_name
