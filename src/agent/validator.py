"""Generic RAG Agent with configurable prompts."""

import logging
from typing import List
from pathlib import Path
import json
from langchain_core.prompts import ChatPromptTemplate
from langchain.schema import Document
from langchain_classic.chains.llm import LLMChain
from src.agent.config_loader import AgentConfig
from src.domain.models import AgentResponse
from src.llm.base import LLMProvider

logger = logging.getLogger(__name__)


class VALIDATORAGENT:
    """Generic RAG agent with configurable behavior."""

    def __init__(
        self,
        config: AgentConfig,
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
        self.llm_provider = llm_provider
        self._setup_chain()

        logger.info(f"Initialized agent: {config.name}")

    def _setup_chain(self) -> None:
        """Set up the RAG chain using configured prompts."""
        prompt = ChatPromptTemplate.from_messages([
            ("system", self.config.system_prompt),
            ("human", self.config.human_prompt),
        ])

        llm = self.llm_provider.get_llm()

        # simple chain — no retriever needed
        self.rag_chain = LLMChain(llm=llm, prompt=prompt,output_key="response")

        logger.info("Validator chain set up (LLM only, no RAG context)")
        
   
    def run(self, input_text: str) -> AgentResponse:
        logger.info(f"Processing input: {input_text[:50]}...")
        try:
            # Get raw LLM output
            response_json = self.rag_chain.invoke({"input": input_text})
            result =  json.loads(response_json["response"])
            logger.info(f"Validator result: {result}'")
            # Parse JSON output
            validity = result.get("valid")

            return AgentResponse(
                input=input_text,
                output=validity,
                source_documents=0,
                error=None if result.get("valid") else result.get("reason", "Invalid input"),
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
