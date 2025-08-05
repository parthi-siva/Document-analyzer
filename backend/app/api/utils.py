from fastapi import UploadFile, HTTPException
from typing import Protocol
import os


# Storage Interface
class StorageInterface(Protocol):
    async def save(self, file: UploadFile, filename: str) -> str:
        """Save file and return the path/identifier"""
        pass


# Local Storage Implementation
class LocalStorage:
    def __init__(self, base_path: str = "./uploads"):
        self.base_path = base_path
        os.makedirs(base_path, exist_ok=True)

    async def save(self, file: UploadFile, filename: str) -> str:
        file_path = os.path.join(self.base_path, filename)
        try:
            with open(file_path, "wb") as buffer:
                content = await file.read()
                buffer.write(content)
            return file_path
        except Exception as e:
            raise HTTPException(
                status_code=500, detail=f"Failed to save file locally: {str(e)}"
            )


# Storage Factory
class StorageFactory:
    @staticmethod
    def create_storage(env: str) -> StorageInterface:
        if env == "development":
            return LocalStorage()
        else:
            raise ValueError("Invalid environment")
