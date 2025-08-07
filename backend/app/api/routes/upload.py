from typing import Annotated
from fastapi import APIRouter, File, HTTPException, Query, UploadFile
import logging
import os

from app.api.utils import StorageFactory
from app.core.service import (
    DocumentParser,
    VectorStore,
)
from app.core.langchain_workflow import LangChainRAGWorkflowOrchestrator
from app.core.langchain_chat_engine import LangChainChatHistoryManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter()
storage = StorageFactory.create_storage(os.getenv("ENVIRONMENT", "development"))

parser = DocumentParser()
vector_store = VectorStore("./chroma_db", "my_documents")
chat_history_manager = LangChainChatHistoryManager(vector_store=vector_store)


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
    session_id: str = Query(
        "default", description="Session ID for conversation history"
    ),
    use_history: bool = Query(True, description="Whether to use conversation history"),
):
    """
    Endpoint to answer a query using the uploaded files.
    """
    uploads_path = "./uploads/"

    try:
        langchain_workflow = LangChainRAGWorkflowOrchestrator(
            parser, vector_store, chat_history_manager
        )

        logger.info("Processing documents from uploads directory")
        result = await langchain_workflow.process_document(uploads_path)
        logger.info("Processed %s documents", result.document_count)

        logger.info(
            f"Querying with: {query} (session: {session_id}, use_history: {use_history})"
        )

        response = await langchain_workflow.query_document_with_history(
            query, session_id
        )

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
