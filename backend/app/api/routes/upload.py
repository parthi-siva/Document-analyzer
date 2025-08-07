from typing import Annotated
from fastapi import APIRouter, File, HTTPException, Query, UploadFile
import logging
import os

from app.api.utils import StorageFactory
from app.core.service import (
    DocumentParser,
    VectorStore,
    RetrievalService,
)
from app.core.langchain_workflow import LangChainRAGWorkflowOrchestrator
from app.core.database import DatabaseChatHistory
from app.core.chat_engine import ChatHistoryManager

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter()
storage = StorageFactory.create_storage(os.getenv("ENVIRONMENT", "development"))

# Initialize services (removed caching as requested)
parser = DocumentParser()
vector_store = VectorStore("./chroma_db", "my_documents")
chat_history_manager = ChatHistoryManager(vector_store=vector_store)
retrieval_service = RetrievalService(vector_store=vector_store)


@router.get("/")
async def root():
    return {"message": "Hello World"}


@router.post("/uploadfiles/")
async def create_upload_files(
    files: Annotated[
        list[UploadFile], File(description="Multiple files as UploadFile")
    ],
):
    """Upload multiple files."""
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

        return {
            "filenames": uploaded_files,
            "message": f"Successfully uploaded {len(uploaded_files)} files",
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

    try:
        # Use the new LangChain-based workflow
        langchain_workflow = LangChainRAGWorkflowOrchestrator(
            parser, vector_store, retrieval_service
        )

        # Process documents from uploads directory
        logger.info("Processing documents from uploads directory")
        result = await langchain_workflow.process_document(uploads_path)
        logger.info("Processed %s documents", result.document_count)

        logger.info(f"Querying with: {query} (session: {session_id}, use_history: {use_history})")

        # Choose between history-aware and simple query based on use_history flag
        response = await langchain_workflow.query_document_with_history(query, session_id)

        logger.info(f"Query response: {response.content[:100]}...")
        return {
            "response": response.content,
            "session_id": session_id,
            "used_history": use_history,
            "source_documents_count": len(response.source_documents),
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
        db_history = DatabaseChatHistory()
        deleted_count = db_history.clear_session_history(session_id)
        logger.info(f"Cleared {deleted_count} messages for session {session_id}")
        return {
            "session_id": session_id,
            "cleared_messages": deleted_count,
            "message": f"Successfully cleared conversation history for session {session_id}"
        }
    except Exception as e:
        logger.error(f"Error clearing conversation history: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
