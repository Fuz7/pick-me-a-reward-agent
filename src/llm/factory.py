"""Factory for creating LLM providers."""

from typing import TYPE_CHECKING

from src.llm.base import LLMProvider
from src.llm.ollama_provider import OllamaProvider
from src.llm.groq_provider import GroqProvider
from src.config.settings import settings

if TYPE_CHECKING:
    from src.agent.config_loader import AgentConfig


class LLMFactory:
    """Factory for creating LLM providers."""

    @staticmethod
    def create(
        model_name: str,
        temperature: float = 0.1,
        provider: str = "ollama",
    ) -> LLMProvider:
        """
        Create an LLM provider.

        Args:
            model_name: Name of the model.
            temperature: Temperature for generation.
            provider: Provider type ("ollama" supported).

        Returns:
            LLMProvider instance.

        Raises:
            ValueError: If provider is not supported.
        """
        if provider == "ollama":
            return OllamaProvider(model=model_name, temperature=temperature)
        elif provider == "groq":                                                  # add this
            return GroqProvider(model=model_name, temperature=temperature,api_key=settings.GROQ_API_KEY) 
        raise ValueError(f"Unsupported LLM provider: {provider}")

    @staticmethod
    def create_from_agent_config(config: "AgentConfig") -> LLMProvider:
        """
        Create an LLM provider from AgentConfig.

        Args:
            config: Agent configuration.

        Returns:
            LLMProvider instance.
        """
        return LLMFactory.create(
            model_name=config.model_name,
            temperature=config.temperature,
            provider=config.model_provider,
        )
