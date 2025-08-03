# Core Services (Separate Concerns)
class DocumentParser:
    async def parse(self, file_path: str) -> DocumentChunks:
        # Handle different file types
        pass

class EmbeddingService:
    async def embed(self, chunks: DocumentChunks) -> Embeddings:
        # Handle batch processing, retries
        pass

class VectorStore:
    async def store(self, embeddings: Embeddings) -> None:
        # Abstract different vector databases
        pass

class RetrievalService:
    async def search(self, query: str) -> RelevantChunks:
        # Hybrid search, reranking logic
        pass

class LLMOrchestrator:
    async def generate(self, context: RelevantChunks, query: str) -> Response:
        # Handle different LLM providers, streaming, caching
        pass
