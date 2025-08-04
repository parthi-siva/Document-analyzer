from typing import Any, Optional, Union, Annotated
import hashlib

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

processed_docs_cache = {}

router = APIRouter()
storage = StorageFactory.create_storage(os.getenv("ENVIRONMENT", "development"))

parser = DocumentParser()
vector_store = VectorStore("./chroma_db", "my_documents")
retrieval_service = RetrievalService(vector_store)
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
    uploads_path = "./uploads/"
    cache_key = get_uploads_hash(uploads_path)

    try:
        workflow = RAGWorkflowOrchestrator(
            parser, vector_store, retrieval_service, llm_orchestrator
        )
        if cache_key in processed_docs_cache:
            logger.info("Using cached results for uploads")
            result = processed_docs_cache[cache_key]
        else:
            result = await workflow.process_document("./uploads/")
            logger.info(f"Processed {result.document_count} documents")
            processed_docs_cache[cache_key] = result
        logger.info(f"Querying with: {query}")
        response = await workflow.query_document(query)
        logger.info(f"Query response: {response.content}")
        return {
            "response": response.content,
        }
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))