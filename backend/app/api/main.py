from fastapi import APIRouter
from app.api.routes import upload
from app.api.utils import StorageFactory

api_router = APIRouter()

api_router.include_router(upload.router)
