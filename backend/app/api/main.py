from fastapi import APIRouter
from app.api.routes import upload
from backend.app.api.utils import StorageFactory
# Removed import of 'story' as it does not exist in 'backend.app.api.routes'

api_router = APIRouter()

api_router.include_router(upload.router)
