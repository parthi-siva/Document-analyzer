from app.core.service import (
    DocumentParser,
    ProcessResult,
    QueryResponse,
    VectorStore,
    RetrievalService,
    LLMOrchestrator,
)

class RAGWorkflowOrchestrator:
    """Main orchestrator that coordinates the RAG pipeline"""
    
    def __init__(self, 
                 document_parser: DocumentParser,
                 vector_store: VectorStore,
                 retrieval_service: RetrievalService,
                 llm_orchestrator: LLMOrchestrator):
        self.document_parser = document_parser
        self.vector_store = vector_store
        self.retrieval_service = retrieval_service
        self.llm_orchestrator = llm_orchestrator
    
    async def process_document(self, file_path: str) -> ProcessResult:
        """Complete pipeline for processing and storing documents"""
        try:
            # Step 1: Parse documents
            documents = await self.document_parser.parse(file_path)
            
            # Step 2: Store documents (embedding + storage happens internally)
            result = await self.vector_store.store(documents)
            
            return result
        except Exception as e:
            raise Exception(f"Document processing pipeline failed: {str(e)}")
    
    async def query_document(self, query: str) -> QueryResponse:
        """Complete pipeline for querying documents and generating responses"""
        try:
            # Step 1: Retrieve relevant context
            retrieval_response = await self.retrieval_service.search(query)
            
            # Step 2: Generate response with LLM
            final_response = await self.llm_orchestrator.generate(
                query, 
                retrieval_response.source_documents
            )
            
            return final_response
        except Exception as e:
            raise Exception(f"Query pipeline failed: {str(e)}")
