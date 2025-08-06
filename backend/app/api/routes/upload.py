from typing import Annotated
import hashlib

from fastapi import APIRouter, File, HTTPException, Query, UploadFile
import logging
import os

from openai import OpenAI
from app.core.custom_llm_wrapper import DeepInfraLLM
from app.api.utils import StorageFactory
from app.core.service import (
    DocumentParser,
    VectorStore,
    RetrievalService,
)
from app.core.cache import CacheFactory
from app.core.config import config
from app.core.orchestrator import LLMOrchestrator
from app.core.workflow import RAGWorkflowOrchestrator
from app.core.database import DatabaseChatHistory
from app.core.chat_engine import ChatHistoryManager

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter()
storage = StorageFactory.create_storage(os.getenv("ENVIRONMENT", "development"))

# Initialize cache for document processing results
processed_docs_cache = CacheFactory.create_cache(
    cache_type=config.cache.CACHE_TYPE,
    default_ttl=config.cache.DOCUMENT_PROCESS_CACHE_TTL,
)

parser = DocumentParser()
vector_store = VectorStore("./chroma_db", "my_documents")
chat_history_manager = ChatHistoryManager(vector_store=vector_store)
retrieval_service = RetrievalService(vector_store=vector_store)
llm_orchestrator = LLMOrchestrator()


def get_uploads_hash(upload_dir: str) -> str:
    """Generate a hash based on the filenames and their modification times in the uploads folder."""
    try:
        files = []
        for fname in sorted(os.listdir(upload_dir)):
            fpath = os.path.join(upload_dir, fname)
            if os.path.isfile(fpath):
                stat = os.stat(fpath)
                files.append(f"{fname}:{stat.st_mtime}")
        hash_str = "|".join(files)
        return hashlib.sha256(hash_str.encode()).hexdigest()
    except Exception as e:
        logger.error(f"Error generating uploads hash: {e}")
        return "no_files"


@router.get("/")
async def root():
    return {"message": "Hello World"}


@router.post("/uploadfiles/")
async def create_upload_files(
    files: Annotated[
        list[UploadFile], File(description="Multiple files as UploadFile")
    ],
):
    """Upload multiple files and invalidate cache when new files are uploaded."""
    if not files:
        logger.error("No files provided for upload")
        raise HTTPException(status_code=400, detail="No files provided")

    uploaded_files = []
    try:
        for file in files:
            if file.filename is None:
                logger.error("File has no filename")
                raise HTTPException(
                    status_code=400, detail="One or more files have no filename"
                )

            file_path = await storage.save(file, file.filename)
            uploaded_files.append(file.filename)
            logger.info(f"Successfully uploaded: {file.filename}")

        # Invalidate cache when new files are uploaded
        uploads_path = "./uploads/"
        new_cache_key = get_uploads_hash(uploads_path)

        # Clear old cache entries if they exist
        await processed_docs_cache.clear()
        logger.info("Cache cleared due to new file uploads")

        return {
            "filenames": uploaded_files,
            "message": f"Successfully uploaded {len(uploaded_files)} files",
            "cache_invalidated": True,
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error during file upload: {e}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")


@router.get("/answer")
async def answer(
    query: str = Query(..., description="Query to answer"),
    top_k: int = Query(5, ge=1, le=10, description="Number of top results to return"),
    session_id: str = Query("default", description="Session ID for conversation history"),
    use_history: bool = Query(True, description="Whether to use conversation history"),
):
    """
    Endpoint to answer a query using the uploaded files.
    """
    uploads_path = "./uploads/"
    cache_key = get_uploads_hash(uploads_path)

    try:
        workflow = RAGWorkflowOrchestrator(
            parser, vector_store, retrieval_service, llm_orchestrator, chat_history_manager
        )

        # Check cache for processed documents
        cached_result = await processed_docs_cache.get(cache_key)
        if cached_result is not None:
            logger.info(
                "Cache HIT: Using cached results for uploads hash: %s...", cache_key[:8]
            )
            result = cached_result
        else:
            logger.info(
                "Cache MISS: Processing documents for hash: %s..., cache_key[:8]"
            )
            result = await workflow.process_document("./uploads/")
            logger.info("Processed %s documents", result.document_count)

            # Store in cache with TTL
            await processed_docs_cache.set(
                cache_key, result, ttl=config.cache.DOCUMENT_PROCESS_CACHE_TTL
            )
        logger.info(f"Querying with: {query} (session: {session_id}, use_history: {use_history})")
        
        # Use conversation-aware query if history is enabled
        if use_history:
            response = await workflow.query_document_with_history(query, session_id)
        else:
            response = await workflow.query_document(query)
            
        logger.info(f"Query response: {response.content}")
        return {
            "response": response.content,
            "session_id": session_id,
            "used_history": use_history,
        }
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/conversation/{session_id}")
async def get_conversation_history(
    session_id: str,
    limit: int = Query(20, ge=1, le=100, description="Maximum number of messages to retrieve")
):
    """Get conversation history for a session."""
    try:
        db_history = DatabaseChatHistory()
        history = db_history.get_history(session_id, limit)
        return {
            "session_id": session_id,
            "history": history,
            "message_count": len(history)
        }
    except Exception as e:
        logger.error(f"Error retrieving conversation history: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/conversation/{session_id}")
async def clear_conversation_history(session_id: str):
    """Clear conversation history for a session."""
    try:
        # Create a new database connection to perform deletion
        db_history = DatabaseChatHistory()
        db = db_history.SessionLocal()
        try:
            from app.core.database import ChatMessage
            deleted_count = db.query(ChatMessage).filter(ChatMessage.session_id == session_id).delete()
            db.commit()
            logger.info(f"Cleared {deleted_count} messages for session {session_id}")
            return {
                "session_id": session_id,
                "cleared_messages": deleted_count,
                "message": f"Successfully cleared conversation history for session {session_id}"
            }
        finally:
            db.close()
    except Exception as e:
        logger.error(f"Error clearing conversation history: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
