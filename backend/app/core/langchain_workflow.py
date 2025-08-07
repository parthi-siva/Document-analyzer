import logging
from app.core.service import (
    DocumentParser,
    ProcessResult,
    QueryResponse,
    VectorStore,
    RetrievalService,
)
from app.core.orchestrator import LLMOrchestrator
from app.core.langchain_chat_engine import LangChainChatHistoryManager

# Configure logging
logger = logging.getLogger(__name__)


class LangChainRAGWorkflowOrchestrator:
    """LangChain-based RAG workflow orchestrator with conversation history support"""
    
    def __init__(
        self,
        document_parser: DocumentParser,
        vector_store: VectorStore,
        retrieval_service: RetrievalService,
    ):
        self.document_parser = document_parser
        self.vector_store = vector_store
        self.retrieval_service = retrieval_service
        # Initialize LangChain chat history manager
        self.langchain_chat_manager = LangChainChatHistoryManager(vector_store)
    
    async def process_document(self, file_path: str) -> ProcessResult:
        """Complete pipeline for processing and storing documents"""
        try:
            logger.info(f"Starting document processing pipeline for: {file_path}")
            
            # Step 1: Parse documents
            documents = await self.document_parser.parse(file_path)
            logger.info(f"Parsed {len(documents)} document chunks")
            
            # Step 2: Store documents (embedding + storage happens internally)
            result = await self.vector_store.store(documents)
            logger.info(f"Successfully stored {result.document_count} documents")
            
            return result
        except Exception as e:
            logger.error(f"Document processing pipeline failed: {str(e)}", exc_info=True)
            raise Exception(f"Document processing pipeline failed: {str(e)}")
    
    async def query_document_with_history(
        self, 
        query: str, 
        session_id: str = "default"
    ) -> QueryResponse:
        """Complete pipeline for querying documents with conversation history awareness using LangChain"""
        try:
            logger.info(f"Starting conversational query: {query[:50]}... (session: {session_id})")
            
            # Use LangChain-based conversational retrieval
            response = await self.langchain_chat_manager.query_with_history(query, session_id)
            
            logger.info("Successfully completed conversational query")
            return response
            
        except Exception as e:
            logger.error(f"Document query with history pipeline failed: {str(e)}", exc_info=True)
            raise Exception(f"Document query with history pipeline failed: {str(e)}")
    
    
    async def get_conversation_history(self, session_id: str, limit: int = 20) -> list:
        """Get conversation history for a session"""
        try:
            logger.info(f"Retrieving conversation history for session: {session_id}")
            
            chat_history = self.langchain_chat_manager.get_session_history(session_id)
            messages = chat_history.messages[-limit:]  # Get last 'limit' messages
            
            # Convert to simple dict format
            history = []
            for msg in messages:
                if hasattr(msg, 'content'):
                    role = "user" if msg.__class__.__name__ == "HumanMessage" else "assistant"
                    history.append({
                        "role": role,
                        "content": msg.content
                    })
            
            logger.info(f"Retrieved {len(history)} messages from history")
            return history
            
        except Exception as e:
            logger.error(f"Failed to retrieve conversation history: {str(e)}", exc_info=True)
            raise Exception(f"Failed to retrieve conversation history: {str(e)}")
