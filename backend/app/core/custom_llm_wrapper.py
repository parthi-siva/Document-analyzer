import logging
from llama_index.core.llms import CustomLLM, CompletionResponse, LLMMetadata
from pydantic import PrivateAttr
from openai import OpenAI


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DeepInfraLLM(CustomLLM):
    """Custom LLM wrapper for OpenAI client with DeepInfra endpoint"""

    _client: OpenAI = PrivateAttr()
    model: str = "Qwen/Qwen3-32B"  # Default model, can be overridden

    def __init__(self, api_key: str | None, model: str = "Qwen/Qwen3-32B"):
        super().__init__()
        self._client = OpenAI(
            api_key=api_key,
            base_url="https://api.deepinfra.com/v1/openai",
        )
        self.model = model
        logger.debug(f"DeepInfraLLM initialized with model: {model}")

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(
            context_window=4096,
            num_output=1024,
            model_name=self.model,
        )

    def complete(self, prompt: str, **kwargs) -> CompletionResponse:
        """Complete using DeepInfra API"""
        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=kwargs.get("temperature", 0.7),
                max_tokens=kwargs.get("max_tokens", 1024),
            )
            return CompletionResponse(
                text=response.choices[0].message.content if response.choices else "",
            )
        except Exception as e:
            logger.error(f"Error in DeepInfraLLM.complete: {e}")
            raise

    def stream_complete(self, prompt: str, **kwargs):
        """Stream completion - not implemented for now"""
        # For simplicity, just return regular completion
        return self.complete(prompt, **kwargs)
