import os
import logging
from typing import List
from dataclasses import dataclass
from pathlib import Path

# LangChain imports
from langchain.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from app.core.custom_embeddings import DeepInfraEmbeddings
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain_openai import ChatOpenAI
import chromadb

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
            # Convert to Path object for better handling
            path = Path(file_path)
            logger.debug(f"Converted path to Path object: {path}")

            # Validate path exists
            if not path.exists():
                logger.error(f"Path does not exist: {file_path}")
                raise FileNotFoundError(f"Path {file_path} does not exist")

            logger.info("Loading documents with DirectoryLoader")
            # Use DirectoryLoader to load documents
            documents = []
            
            # Load different file types
            for file_pattern in ["*.txt", "*.md", "*.pdf"]:
                try:
                    if file_pattern == "*.pdf":
                        loader = DirectoryLoader(
                            str(path), 
                            glob=file_pattern,
                            loader_cls=PyPDFLoader,
                            show_progress=True
                        )
                    else:
                        loader = DirectoryLoader(
                            str(path), 
                            glob=file_pattern,
                            loader_cls=TextLoader,
                            show_progress=True
                        )
                    file_docs = loader.load()
                    documents.extend(file_docs)
                except Exception as e:
                    logger.warning(f"Could not load files with pattern {file_pattern}: {e}")
            
            logger.info(f"Successfully loaded {len(documents)} documents")
            
            # Split documents into chunks
            logger.info("Splitting documents into chunks")
            chunked_documents = self.text_splitter.split_documents(documents)
            logger.info(f"Created {len(chunked_documents)} chunks")

            return chunked_documents
        except Exception as e:
            logger.error(f"Failed to parse documents: {str(e)}", exc_info=True)
            raise Exception(f"Failed to parse documents: {str(e)}")


class EmbeddingService:
    """Service responsible for generating embeddings from document chunks"""

    def __init__(self):
        """
        Initialize with OpenAI embeddings via DeepInfra
        """
        logger.info("Initializing EmbeddingService with OpenAI embeddings")
        api_key = os.environ.get("OPENAI_API_KEY", "")
        if not api_key:
            logger.warning("OPENAI_API_KEY environment variable is not set")
        
        # Use DeepInfra embeddings
        self.embedding_model = DeepInfraEmbeddings(
            api_key=api_key,
            model="text-embedding-3-small",  # You can change this to your preferred embedding model
        )
        logger.debug("OpenAI embedding model initialized successfully")


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
            # Add documents to the vector store
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
                search_type="similarity",
                search_kwargs={"k": k}
            )
            logger.info("Retriever created successfully")
            return retriever
        except Exception as e:
            logger.error(f"Failed to create retriever: {str(e)}", exc_info=True)
            raise Exception(f"Failed to create retriever: {str(e)}")


class RetrievalService:
    """Service responsible for retrieving relevant documents based on queries"""

    def __init__(self, vector_store: VectorStore):
        logger.info("Initializing RetrievalService for document retrieval")
        self.vector_store = vector_store
        logger.debug("RetrievalService initialized successfully")

    async def search(self, query: str, top_k: int = 5) -> List[Document]:
        """Search for relevant documents based on query (retrieval only, no LLM generation)"""
        logger.info(f"Starting document retrieval with query: {query[:50]}... top_k: {top_k}")
        try:
            logger.debug("Getting retriever from vector store")
            # Get retriever from vector store
            retriever = self.vector_store.get_retriever(k=top_k)

            logger.info("Executing document retrieval (no LLM generation)")
            # Retrieve documents without LLM generation
            retrieved_documents = retriever.invoke(query)
            logger.debug(f"Retrieved {len(retrieved_documents)} documents")

            logger.info("Document retrieval completed successfully")
            return retrieved_documents
        except Exception as e:
            logger.error(f"Failed to retrieve documents: {str(e)}", exc_info=True)
            raise Exception(f"Failed to retrieve documents: {str(e)}")
