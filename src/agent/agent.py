"""Generic RAG Agent with configurable prompts."""

import logging
from typing import List

from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

from src.agent.config_loader import AgentConfig
from src.domain.models import AgentResponse
from src.repositories.base import VectorStoreRepository
from src.llm.base import LLMProvider

logger = logging.getLogger(__name__)


class RAGAgent:
    """Generic RAG agent with configurable behavior."""

    def __init__(
        self,
        config: AgentConfig,
        repository: VectorStoreRepository,
        llm_provider: LLMProvider,
    ):
        """
        Initialize the RAG agent with injected dependencies.

        Args:
            config: Agent configuration from YAML.
            repository: Vector store repository for retrieval.
            llm_provider: LLM provider for generation.
        """
        self.config = config
        self.repository = repository
        self.llm_provider = llm_provider
        self._setup_chain()

        logger.info(f"Initialized agent: {config.name}")

    def _setup_chain(self) -> None:
        """Set up the RAG chain using configured prompts."""
        prompt = ChatPromptTemplate.from_messages([
            ("system", self.config.system_prompt),
            ("human", self.config.human_prompt),
        ])

        retriever = self.repository.as_retriever(k=self.config.retriever_k)
        llm = self.llm_provider.get_llm()

        question_answer_chain = create_stuff_documents_chain(llm, prompt)
        self.rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    def run(self, input_text: str) -> AgentResponse:
        """
        Run the agent on a single input.

        Args:
            input_text: The input text to process.

        Returns:
            AgentResponse with the result.
        """
        logger.info(f"Processing input: {input_text[:50]}...")
        try:
            response = self.rag_chain.invoke({"input": input_text})
            return AgentResponse(
                input=input_text,
                output=response["answer"],
                source_documents=len(response.get("context", [])),
            )
        except Exception as e:
            logger.error(f"Error processing input: {e}")
            return AgentResponse(
                input=input_text,
                output="",
                source_documents=0,
                error=str(e),
            )

    def run_batch(self, inputs: List[str]) -> List[AgentResponse]:
        """
        Run the agent on multiple inputs.

        Args:
            inputs: List of input texts to process.

        Returns:
            List of AgentResponse objects.
        """
        return [self.run(input_text) for input_text in inputs]

    def run_test_cases(self) -> List[AgentResponse]:
        """
        Run the agent on configured test cases.

        Returns:
            List of AgentResponse objects for each test case.
        """
        if not self.config.test_cases:
            logger.warning("No test cases configured")
            return []

        logger.info(f"Running {len(self.config.test_cases)} test cases...")
        return self.run_batch(self.config.test_cases)