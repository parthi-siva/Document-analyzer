from app.core.service import (
    DocumentParser,
    ProcessResult,
    QueryResponse,
    VectorStore,
    RetrievalService,
)
from app.core.orchestrator import LLMOrchestrator
from app.core.chat_engine import ChatHistoryManager
from app.services.prompt_manager import get_question_answer_prompt


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

    async def query_document(self, query: str) -> QueryResponse:
        """Complete pipeline for querying documents and generating responses"""
        try:
            # Step 1: Retrieve relevant context (no LLM call here)
            source_documents = await self.retrieval_service.search(query)

            # Step 2: Generate response with LLM (single API call)
            final_response = await self.llm_orchestrator.generate(
                query, source_documents
            )

            return final_response
        except Exception as e:
            raise Exception(f"Query pipeline failed: {str(e)}")

    async def query_document_with_history(self, query: str, session_id: str = "default") -> QueryResponse:
        """Complete pipeline for querying documents with conversation history awareness"""
        try:
            # Step 1: Create chat engine with conversation history
            chat_engine, _ = self.chat_history_manager.create_chat_engine_with_history(session_id)
            query = query.strip()
            
            _, user_prompt = get_question_answer_prompt(context="", query=query)

            # Step 3: Generate response with LLM using chat engine (includes conversation context)
            chat_response = chat_engine.chat(user_prompt)

            # Step 4: Save the conversation to database
            self.chat_history_manager.save_chat_interaction(session_id, user_prompt, str(chat_response))

            # Step 5: Convert to QueryResponse format
            final_response = QueryResponse(
                content=str(chat_response),
                source_documents=[],
                metadata={"model": "Qwen/Qwen3-32B"},
            )

            return final_response
        except Exception as e:
            raise Exception(f"Query with history pipeline failed: {str(e)}")
