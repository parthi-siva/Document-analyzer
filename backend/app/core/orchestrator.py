import os
import logging
from typing import List, Optional
from llama_index.core.schema import Document
from llama_index.core.base.llms.base import BaseLLM
from openai import OpenAI

from app.core.service import QueryResponse

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LLMOrchestrator:
    """Service responsible for generating responses using LLM"""

    def __init__(self, llm: Optional[BaseLLM] = None):
        logger.info("Initializing LLMOrchestrator with DeepInfra OpenAI client")
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            logger.warning(
                "OPENAI_API_KEY environment variable is not set - LLM calls may fail"
            )

        self.llm = OpenAI(
            api_key=api_key,
            base_url="https://api.deepinfra.com/v1/openai",
        )
        logger.debug("OpenAI client initialized with DeepInfra endpoint")

    async def generate(self, query: str, context: List[Document]) -> QueryResponse:
        """Generate response using LLM with provided context"""
        logger.info(f"Starting LLM response generation for query: {query[:50]}...")
        try:
            logger.debug(f"Formatting context from {len(context)} documents")
            # Format context for prompt
            context_text = "\n\n".join(
                [doc.text for doc in context[:3]]
            )  # Limit context
            logger.debug(f"Context text length: {len(context_text)} characters")

            system_prompt = "You are a helpful assistant that answers questions based on provided context."
            # Create prompt with context
            prompt = f"""
            Context information is below.
            ---------------------
            {context_text}
            ---------------------
            Given the context information and not prior knowledge, 
            answer the query.
            Query: {query}
            Answer:
            """
            logger.debug(f"Prompt length: {len(prompt)} characters")

            logger.info("Calling DeepInfra LLM API")
            # Generate response
            response = self.llm.chat.completions.create(
                model="meta-llama/Llama-2-70b-chat-hf",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.7,
                top_p=0.8,
                max_tokens=1024,
            )
            logger.debug("LLM API call completed successfully")
            content = response.choices[0].message.content
            if not content:
                logger.warning("LLM response is empty, returning default message")
                content = "No relevant information found in the provided context."
            result = QueryResponse(
                content=content,
                source_documents=context,
                metadata={"model": "Qwen/Qwen3-32B", "prompt_length": len(prompt)},
            )
            logger.info("LLM response generation completed successfully")
            return result
        except Exception as e:
            logger.error(f"Failed to generate response: {str(e)}", exc_info=True)
            raise Exception(f"Failed to generate response: {str(e)}")
