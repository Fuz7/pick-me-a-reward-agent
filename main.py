"""
Generic RAG Agent

A configurable RAG system that loads behavior from agent.yaml.
Supports custom prompts, different LLM providers, and flexible document processing.

Usage:
    python main.py                    # Uses default agent.yaml
    python main.py --config my.yaml   # Uses custom config file
"""

import argparse
import logging

from src.agent import AgentConfig, RAGAgent
from src.loaders import DocumentLoaderFactory
from src.chunkers import ChunkerFactory
from src.repositories import ChromaRepository
from src.embeddings import EmbeddingFactory
from src.llm import LLMFactory
from src.config.settings import settings


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main(config_path: str = "agent.yaml"):
    """Main execution function."""
    # Load configuration from YAML
    config = AgentConfig.from_yaml(config_path)
    logging.info(settings.MODEL_PROVIDER)
    logger.info(f"Starting agent: {config.name}")
    logger.info(f"Using model: {config.model_provider}/{config.model_name}")

    try:
        # Create embedding provider (Factory pattern)
        embedding_provider = EmbeddingFactory.create_from_agent_config(config)
        embeddings = embedding_provider.get_embeddings()

        # Create repository (Repository pattern)
        repository = ChromaRepository(
            persist_dir=config.persist_dir,
            embeddings=embeddings,
        )

        # Load or build vector store
        if not repository.load():
            logger.info("Building new vector database...")

            # Load documents (Factory + Strategy pattern)
            file_path, documents = DocumentLoaderFactory.detect_and_load(
                config.context_file
            )

            # Chunk documents (Factory + Strategy pattern)
            chunker = ChunkerFactory.create_from_agent_config(file_path, config)
            chunks = chunker.chunk(documents)

            # Save to repository
            repository.save(chunks)
            logger.info("Vector database created successfully")

        # Create LLM provider (Factory pattern)
        llm_provider = LLMFactory.create_from_agent_config(config)

        # Create agent with injected dependencies
        agent = RAGAgent(
            config=config,
            repository=repository,
            llm_provider=llm_provider,
        )

        # Run test cases
        print("\n" + "=" * 80)
        print(f"{config.name.upper()}")
        print(f"{config.description}")
        print("=" * 80 + "\n")

        results = agent.run_test_cases()

        for i, result in enumerate(results, 1):
            print(f"\n{'─' * 80}")
            print(f"Test Case {i}:")
            print(f"{'─' * 80}")
            print(f"Input: {result.input}")
            if result.is_success:
                print(f"\nOutput:\n{result.output}")
                print(f"\nSources used: {result.source_documents} document chunks")
            else:
                print(f"\nError: {result.error}")

        print("\n" + "=" * 80 + "\n")

    except FileNotFoundError as e:
        logger.error(str(e))
        print(f"\nError: {e}")

    except Exception as e:
        logger.error(f"An error occurred: {e}", exc_info=True)
        print(f"\nError: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run RAG Agent")
    parser.add_argument(
        "--config",
        "-c",
        default="agent.yaml",
        help="Path to agent configuration file (default: agent.yaml)",
    )
    args = parser.parse_args()

    main(args.config)