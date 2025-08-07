import logging
from app.core.service import (
    DocumentParser,
    ProcessResult,
    QueryResponse,
    VectorStore,
    RetrievalService,
)
from app.core.orchestrator import LLMOrchestrator
from app.core.chat_engine import ChatHistoryManager
from app.core.langchain_chat_engine import LangChainChatHistoryManager
from app.services.prompt_manager import get_question_answer_prompt

# Configure logging
logger = logging.getLogger(__name__)


class RAGWorkflowOrchestrator:
    """Main orchestrator that coordinates the RAG pipeline"""

    def __init__(
        self,
        document_parser: DocumentParser,
        vector_store: VectorStore,
        retrieval_service: RetrievalService,
        llm_orchestrator: LLMOrchestrator,
        chat_history_manager: ChatHistoryManager,
    ):
        self.document_parser = document_parser
        self.vector_store = vector_store
        self.retrieval_service = retrieval_service
        self.llm_orchestrator = llm_orchestrator
        self.chat_history_manager = chat_history_manager or ChatHistoryManager(
            vector_store=self.vector_store
        )

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

    async def query_document_with_history(self, query: str, session_id: str = "default") -> QueryResponse:
        """Complete pipeline for querying documents with conversation history awareness"""
        try:
            # Step 1: Create chat engine with history
            chat_engine, memory = self.chat_history_manager.create_chat_engine_with_history(session_id)
            
            # Step 2: Get the response with conversation context
            response = await chat_engine.achat(query)
            
            # Step 3: Save the interaction
            self.chat_history_manager.save_chat_interaction(session_id, query, str(response))
            
            # Step 4: Format the response
            query_response = QueryResponse(
                content=str(response),
                source_documents=[],  # Chat engine handles this internally
                metadata={"session_id": session_id, "has_history": True}
            )
            
            return query_response
        except Exception as e:
            raise Exception(f"Document query with history pipeline failed: {str(e)}")
    
    async def query_document(self, query: str, top_k: int = 5) -> QueryResponse:
        """Complete pipeline for querying documents without conversation history"""
        try:
            # Step 1: Retrieve relevant documents
            documents = await self.retrieval_service.search(query, top_k)
            
            # Step 2: Generate response using LLM
            response = await self.llm_orchestrator.generate(query, documents)
            
            return response
        except Exception as e:
            raise Exception(f"Document query pipeline failed: {str(e)}")
