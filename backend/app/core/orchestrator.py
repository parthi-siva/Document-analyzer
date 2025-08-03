class RAGWorkflow:
    def __init__(self):
        self.parser = DocumentParser()
        self.embedder = EmbeddingService()
        self.vector_store = VectorStore()
        self.retriever = RetrievalService()
        self.llm = LLMOrchestrator()

    async def process_document(self, file_path: str) -> ProcessResult:
        # Step 1: Parse document
        chunks = await self.parser.parse(file_path)

        # Step 2: Generate embeddings
        embeddings = await self.embedder.embed(chunks)

        # Step 3: Store in vector database
        await self.vector_store.store(embeddings)

        return ProcessResult(success=True)

    async def query_document(self, query: str) -> QueryResponse:
        # Step 1: Retrieve relevant context
        context = await self.retriever.search(query)

        # Step 2: Generate response
        response = await self.llm.generate(context, query)

        return QueryResponse(content=response)
