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

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main(config_path: str = "agent.yaml", validator_config_path: str = "validator.yaml"):
    """Main execution function."""
    # Load configuration from YAML
    config = AgentConfig.from_yaml(config_path)

    validator = None
    if validator_config_path:
        validator_config = AgentConfig.from_yaml(validator_config_path)
        validator_llm = LLMFactory.create_from_agent_config(validator_config)

        validator = RAGAgent(
            config=validator_config,
            repository=None,
            llm_provider=validator_llm,
        )
    
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

        results = []

        for test_input in config.test_cases:
            # Run validator first (if exists)
            if validator:
                validation_result = validator.run(test_input)

                if not validation_result.is_success:
                    results.append(validation_result)
                    continue

                # Expect JSON like: {"valid": true, "reason": "..."}
                validation_output = validation_result.output.lower()

                if "false" in validation_output:
                    results.append(
                        validation_result
                    )
                    continue

            # If valid → run reward agent
            result = agent.run(test_input)
            results.append(result)
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
    parser.add_argument(
    "--validator-config",
    "-v",
    default="validator.yaml",
    help="Validator system config file"
    )
    args = parser.parse_args()
    main(args.config, args.validator_config)
