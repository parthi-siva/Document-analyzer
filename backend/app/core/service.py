import os
import logging
from typing import List, Optional
from dataclasses import dataclass
from pathlib import Path
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.schema import Document
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import StorageContext
from llama_index.core.base.llms.base import BaseLLM
import chromadb
from openai import OpenAI

from app.core.custom_embeddings import DeepInfraEmbeddingModel

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

    async def parse(self, file_path: str) -> List[Document]:
        """Parse documents from directory path"""
        logger.info(f"Starting document parsing for path: {file_path}")
        try:
            # Convert to Path object for better handling
            path = Path(file_path)
            logger.debug(f"Converted path to Path object: {path}")

            # Validate path exists
            if not path.exists():
                logger.error(f"Path does not exist: {file_path}")
                raise FileNotFoundError(f"Path {file_path} does not exist")

            logger.info("Loading documents with SimpleDirectoryReader")
            # Use SimpleDirectoryReader to load documents
            reader = SimpleDirectoryReader(
                input_dir=str(path) if path.is_dir() else str(path.parent),
                filename_as_id=True,
            )
            documents = reader.load_data()
            logger.info(f"Successfully loaded {len(documents)} documents")

            return documents
        except Exception as e:
            logger.error(f"Failed to parse documents: {str(e)}", exc_info=True)
            raise Exception(f"Failed to parse documents: {str(e)}")


class EmbeddingService:
    """Service responsible for generating embeddings from document chunks"""

    def __init__(self):
        """
        Initialize with HuggingFace embedding model
        For DeepInfra, we can use open-source embedding models locally
        """
        logger.info("Initializing EmbeddingService with DeepInfraEmbeddingModel")
        api_key = os.environ.get("OPENAI_API_KEY", "")
        if not api_key:
            logger.warning("OPENAI_API_KEY environment variable is not set")
        self.embedding_model = DeepInfraEmbeddingModel(api_key=api_key)
        logger.debug("DeepInfraEmbeddingModel initialized successfully")


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
        logger.debug("Creating ChromaDB PersistentClient")
        self.client = chromadb.PersistentClient(path=persist_path)
        logger.debug("Getting or creating ChromaDB collection")
        self.chroma_collection = self.client.get_or_create_collection(collection_name)
        logger.debug("Initializing ChromaVectorStore")
        self.vector_store = ChromaVectorStore(chroma_collection=self.chroma_collection)
        logger.info("VectorStore initialized successfully")

    async def store(self, documents: List[Document]) -> ProcessResult:
        """Store documents with embeddings in vector database"""
        logger.info(f"Starting storage of {len(documents)} documents")
        try:
            logger.debug("Initializing EmbeddingService")
            # Use local embedding model for storage
            embedding_service = EmbeddingService()
            logger.debug("Creating StorageContext")
            storage_context = StorageContext.from_defaults(
                vector_store=self.vector_store
            )

            logger.info("Creating VectorStoreIndex from documents")
            # Create index which will automatically store in ChromaDB
            index = VectorStoreIndex.from_documents(
                documents,
                storage_context=storage_context,
                embed_model=embedding_service.embedding_model,
            )
            logger.info("VectorStoreIndex created successfully")

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

    def get_index(self) -> VectorStoreIndex:
        """Retrieve existing index from vector store"""
        logger.info("Retrieving existing VectorStoreIndex")
        try:
            logger.debug("Initializing EmbeddingService for querying")
            # Use the same embedding model for querying as we used for indexing
            embedding_service = EmbeddingService()
            logger.debug("Creating StorageContext for retrieval")
            storage_context = StorageContext.from_defaults(
                vector_store=self.vector_store
            )
            logger.info("Creating VectorStoreIndex from vector store")
            # This assumes documents were already stored
            index = VectorStoreIndex.from_vector_store(
                vector_store=self.vector_store,
                storage_context=storage_context,
                embed_model=embedding_service.embedding_model,
            )
            logger.info("VectorStoreIndex retrieved successfully")
            return index
        except Exception as e:
            logger.error(f"Failed to retrieve index: {str(e)}", exc_info=True)
            raise Exception(f"Failed to retrieve index: {str(e)}")


class RetrievalService:
    """Service responsible for retrieving relevant documents based on queries"""

    def __init__(self, vector_store: VectorStore):
        logger.info("Initializing RetrievalService for document retrieval only")
        self.vector_store = vector_store
        logger.debug("RetrievalService initialized successfully")

    async def search(self, query: str, top_k: int = 5) -> List[Document]:
        """Search for relevant documents based on query (retrieval only, no LLM generation)"""
        logger.info(f"Starting document retrieval with query: {query[:50]}... top_k: {top_k}")
        try:
            logger.debug("Retrieving VectorStoreIndex")
            # Get existing index
            index = self.vector_store.get_index()

            logger.debug("Creating retriever for document search")
            # Create retriever instead of query engine to avoid LLM call
            retriever = index.as_retriever(similarity_top_k=top_k)

            logger.info("Executing document retrieval (no LLM generation)")
            # Retrieve documents without LLM generation
            retrieved_nodes = retriever.retrieve(query)
            logger.debug(f"Retrieved {len(retrieved_nodes)} document nodes")

            # Extract source documents
            logger.debug("Extracting source documents from nodes")
            source_documents = [node.node for node in retrieved_nodes]
            logger.info(f"Extracted {len(source_documents)} source documents")

            logger.info("Document retrieval completed successfully")
            return source_documents
        except Exception as e:
            logger.error(f"Failed to retrieve documents: {str(e)}", exc_info=True)
            raise Exception(f"Failed to retrieve documents: {str(e)}")
