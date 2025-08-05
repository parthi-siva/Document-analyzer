from typing import List
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.embeddings import BaseEmbedding
from openai import OpenAI
from pydantic import PrivateAttr


class DeepInfraEmbeddingModel(BaseEmbedding):
    _client: OpenAI = PrivateAttr()

    def __init__(self, api_key: str, **kwargs):
        super().__init__(**kwargs)
        base_url = "https://api.deepinfra.com/v1/openai"
        print(f"Initializing DeepInfra client with base_url: {base_url}")
        self._client = OpenAI(
            api_key=api_key,
            base_url=base_url,
        )

    def _get_query_embedding(self, query: str) -> List[float]:
        print(f"Making embedding request to DeepInfra for query: {query[:50]}...")
        try:
            response = self._client.embeddings.create(
                model="Qwen/Qwen3-Embedding-0.6B",
                input=[query],
                encoding_format="float",
            )
            print(f"Successfully received embedding response from DeepInfra")
            return response.data[0].embedding
        except Exception as e:
            print(f"Error calling DeepInfra embedding API: {e}")
            raise

    def _get_text_embedding(self, text: str) -> List[float]:
        response = self._client.embeddings.create(
            model="Qwen/Qwen3-Embedding-0.6B", input=[text], encoding_format="float"
        )
        return response.data[0].embedding

    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        response = self._client.embeddings.create(
            model="Qwen/Qwen3-Embedding-0.6B", input=texts, encoding_format="float"
        )
        return [embedding.embedding for embedding in response.data]

    # Async versions
    async def _aget_query_embedding(self, query: str) -> List[float]:
        return self._get_query_embedding(query)

    async def _aget_text_embedding(self, text: str) -> List[float]:
        return self._get_text_embedding(text)

    async def _aget_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        return self._get_text_embeddings(texts)
