"""Groq LLM provider implementation."""

import logging
from typing import Any

from langchain_groq import ChatGroq

from src.llm.base import LLMProvider

logger = logging.getLogger(__name__)


class GroqProvider(LLMProvider):
    """LLM provider using Groq API."""

    def __init__(self, model: str = "llama-3.3-70b-versatile", temperature: float = 0.1, api_key: str | None = None):
        """
        Initialize the Groq provider.

        Args:
            model: Groq model name.
            temperature: Temperature for generation.
            api_key: Groq API key. If None, reads from GROQ_API_KEY env var.
        """
        self._model = model
        self._temperature = temperature
        self._llm = ChatGroq(model=model, temperature=temperature, api_key=api_key)
        logger.info(f"Initialized Groq provider with model: {model}")

    def get_llm(self) -> Any:
        """Get the ChatGroq instance."""
        return self._llm

    @property
    def model_name(self) -> str:
        """Get the model name."""
        return self._model