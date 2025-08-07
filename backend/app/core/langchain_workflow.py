import logging
from app.core.service import (
    DocumentParser,
    ProcessResult,
    QueryResponse,
    VectorStore,
)
from app.core.langchain_chat_engine import LangChainChatHistoryManager

logger = logging.getLogger(__name__)


class LangChainRAGWorkflowOrchestrator:
    """LangChain-based RAG workflow orchestrator with conversation history support"""

    def __init__(
        self,
        document_parser: DocumentParser,
        vector_store: VectorStore,
        langchain_chat_manager: LangChainChatHistoryManager,
    ):
        self.document_parser = document_parser
        self.vector_store = vector_store
        self.langchain_chat_manager = langchain_chat_manager

    async def process_document(self, file_path: str) -> ProcessResult:
        """Complete pipeline for processing and storing documents"""
        try:
            logger.info(f"Starting document processing pipeline for: {file_path}")

            documents = await self.document_parser.parse(file_path)
            logger.info(f"Parsed {len(documents)} document chunks")

            result = await self.vector_store.store(documents)
            logger.info(f"Successfully stored {result.document_count} documents")

            return result
        except Exception as e:
            logger.error(
                f"Document processing pipeline failed: {str(e)}", exc_info=True
            )
            raise Exception(f"Document processing pipeline failed: {str(e)}")

    async def query_document_with_history(
        self, query: str, session_id: str = "default"
    ) -> QueryResponse:
        """Complete pipeline for querying documents with conversation history awareness using LangChain"""
        try:
            logger.info(
                f"Starting conversational query: {query[:50]}... (session: {session_id})"
            )

            response = await self.langchain_chat_manager.query_with_history(
                query, session_id
            )

            logger.info("Successfully completed conversational query")
            return response

        except Exception as e:
            logger.error(
                f"Document query with history pipeline failed: {str(e)}", exc_info=True
            )
            raise Exception(f"Document query with history pipeline failed: {str(e)}")
