import os
import logging
from typing import List
from dataclasses import dataclass
from pathlib import Path

# LangChain imports
from langchain_community.document_loaders import (
    DirectoryLoader,
    PyPDFLoader,
)
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from pydantic import SecretStr

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class QueryResponse:
    content: str
    source_documents: List[Document]
    metadata: dict


@dataclass
class ProcessResult:
    success: bool
    document_count: int
    chunk_count: int
    metadata: dict


class DocumentParser:
    """Service responsible for parsing documents from various formats"""

    def __init__(self):
        logger.info("Initializing DocumentParser")
        # Initialize text splitter for chunking
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            is_separator_regex=False,
        )

    async def parse(self, file_path: str) -> List[Document]:
        """Parse documents from directory path"""
        logger.info(f"Starting document parsing for path: {file_path}")
        try:
            path = Path(file_path)
            logger.debug(f"Converted path to Path object: {path}")

            if not path.exists():
                logger.error(f"Path does not exist: {file_path}")
                raise FileNotFoundError(f"Path {file_path} does not exist")

            logger.info("Loading documents with DirectoryLoader")
            documents = []

            try:
                loader = DirectoryLoader(
                    str(path), glob="*.pdf", loader_cls=PyPDFLoader, show_progress=True
                )
                file_docs = loader.load()
                documents.extend(file_docs)
            except Exception as e:
                logger.warning(f"Could not load files: {e}")

            logger.info(f"Successfully loaded {len(documents)} documents")

            logger.info("Splitting documents into chunks")
            chunked_documents = self.text_splitter.split_documents(documents)
            logger.info(f"Created {len(chunked_documents)} chunks")

            return chunked_documents
        except Exception as e:
            logger.error(f"Failed to parse documents: {str(e)}", exc_info=True)
            raise Exception(f"Failed to parse documents: {str(e)}")


class EmbeddingService:
    """Service responsible for generating embeddings from document chunks"""
    _instance = None
    _embedding_model = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(EmbeddingService, cls).__new__(cls)
            cls._initialize_embedding_model()
        return cls._instance

    @classmethod
    def _initialize_embedding_model(cls):
        """
        Initialize with Sentence Transformers embeddings (local, no API key needed)
        This method is called only once when the singleton is first created
        """
        if cls._embedding_model is not None:
            return
            
        logger.info("Initializing EmbeddingService with Sentence Transformers (singleton)")
        from langchain_community.embeddings import HuggingFaceEmbeddings

        cls._embedding_model = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": False},
        )
        logger.info(
            "Sentence Transformers embedding model initialized successfully (singleton)"
        )

    @property
    def embedding_model(self):
        """Get the embedding model instance"""
        return self._embedding_model
class VectorStore:
    """Service responsible for storing and retrieving embeddings from ChromaDB"""

    def __init__(
        self, persist_path: str = "./chroma_db", collection_name: str = "rag_collection"
    ):
        logger.info(
            f"Initializing VectorStore with path: {persist_path}, collection: {collection_name}"
        )
        self.persist_path = persist_path
        self.collection_name = collection_name
        logger.debug("Initializing EmbeddingService")
        self.embedding_service = EmbeddingService()
        logger.debug("Creating Chroma vector store")

        # Initialize Chroma vector store
        self.vector_store = Chroma(
            collection_name=collection_name,
            embedding_function=self.embedding_service.embedding_model,
            persist_directory=persist_path,
        )
        logger.info("VectorStore initialized successfully")

    async def store(self, documents: List[Document]) -> ProcessResult:
        """Store documents with embeddings in vector database"""
        logger.info(f"Starting storage of {len(documents)} documents")
        try:
            logger.info("Adding documents to Chroma vector store")

            self.vector_store.add_documents(documents)
            logger.info("Documents added to vector store successfully")

            result = ProcessResult(
                success=True,
                document_count=len(documents),
                chunk_count=len(documents),
                metadata={
                    "collection_name": self.collection_name,
                    "storage_path": self.persist_path,
                },
            )
            logger.info(
                f"Storage completed successfully. Document count: {len(documents)}"
            )
            return result
        except Exception as e:
            logger.error(f"Failed to store embeddings: {str(e)}", exc_info=True)
            raise Exception(f"Failed to store embeddings: {str(e)}")

    def get_retriever(self, k: int = 5):
        """Get retriever from vector store"""
        logger.info("Creating retriever from vector store")
        try:
            # Create retriever with similarity search
            retriever = self.vector_store.as_retriever(
                search_type="similarity", search_kwargs={"k": k}
            )
            logger.info("Retriever created successfully")
            return retriever
        except Exception as e:
            logger.error(f"Failed to create retriever: {str(e)}", exc_info=True)
            raise Exception(f"Failed to create retriever: {str(e)}")
