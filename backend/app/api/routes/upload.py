from typing import Any, Optional, Union, Annotated

from fastapi import APIRouter, File, HTTPException, Query, UploadFile
import logging
import os
from app.api.utils import StorageFactory
from app.core.service import (
    DocumentParser,
    VectorStore,
    RetrievalService,
    LLMOrchestrator,
)
from app.core.orchestrator import RAGWorkflowOrchestrator
# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter()
storage = StorageFactory.create_storage(os.getenv("ENVIRONMENT", "development"))

@router.get("/")
async def root():
    return {"message": "Hello World"}

@router.post("/uploadfiles/")
async def create_upload_files(
        files: Annotated[
        list[UploadFile], File(description="Multiple files as UploadFile")
    ],
):
    if not files:
        logger.error("No files provided for upload")
        raise HTTPException(status_code=400, detail="No files provided")
    for file in files:
        if file.filename is None:
            logger.error("File has no filename")
            raise HTTPException(status_code=400, detail="One or more files have no filename")
        await storage.save(file, file.filename)
    return {"filenames": [file.filename for file in files]}

@router.get("/answer")
async def answer(
        query: str = Query(..., description="Query to answer"),
        top_k: int = Query(5, ge=1, le=10, description="Number of top results to return")
):
    """
    Endpoint to answer a query using the uploaded files.
    """
    parser = DocumentParser()
    vector_store = VectorStore("./chroma_db", "my_documents")
    retrieval_service = RetrievalService(vector_store)
    llm_orchestrator = LLMOrchestrator()

    workflow = RAGWorkflowOrchestrator(
        parser, vector_store, retrieval_service, llm_orchestrator
    )

    try:
        result = await workflow.process_document("./uploads/")
        logger.info(f"Processed {result.document_count} documents")
        logger.info(f"Querying with: {query}")
        response = await workflow.query_document(query)
        logger.info(f"Query response: {response.content}")
        return {
            "response": response.content,
        }
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))