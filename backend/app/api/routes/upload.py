from typing import Any, Optional, Union, Annotated

from fastapi import APIRouter, File, HTTPException, Query, UploadFile
import logging
import os
from app.api.utils import StorageFactory

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
