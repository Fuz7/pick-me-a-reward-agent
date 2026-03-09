"""
Cached RAG Agent Loader for Streamlit

Builds the same RAG agent used in main.py but returns the agent
instead of running test cases. Cached to avoid rebuilding vector DB.
"""

import streamlit as st
import logging
from langchain_core.output_parsers import JsonOutputParser

from src.agent import AgentConfig, RAGAgent
from src.loaders import DocumentLoaderFactory
from src.chunkers import ChunkerFactory
from src.repositories import ChromaRepository
from src.embeddings import EmbeddingFactory
from src.llm import LLMFactory
import os
logger = logging.getLogger(__name__)


@st.cache_resource
def load_rag_agent(config_path: str = "agent.yaml") -> RAGAgent:
    """
    Load and cache the full RAG agent.

    Args:
        config_path (str): Path to YAML config

    Returns:
        RAGAgent
    """

    # Load configuration
    config = AgentConfig.from_yaml(config_path)
    os.environ["GROQ_API_KEY"] = st.secrets.get("GROQ_API_KEY", "")  #
    logger.info(f"Loading agent: {config.name}")

    # Create embeddings
    embedding_provider = EmbeddingFactory.create_from_agent_config(config)
    embeddings = embedding_provider.get_embeddings()

    # Create vector repository
    repository = ChromaRepository(
        persist_dir=config.persist_dir,
        embeddings=embeddings,
    )

    # Load or build vector DB
    if not repository.load():
        logger.info("Building new vector database...")

        file_path, documents = DocumentLoaderFactory.detect_and_load(
            config.context_file
        )

        chunker = ChunkerFactory.create_from_agent_config(file_path, config)
        chunks = chunker.chunk(documents)

        repository.save(chunks)

        logger.info("Vector database built")

    # Create LLM
    llm_provider = LLMFactory.create_from_agent_config(config)

    # Create agent
    agent = RAGAgent(
        config=config,
        repository=repository,
        llm_provider=llm_provider,
    )

    return agent