from typing import List, Optional, Any
from dataclasses import dataclass
from pathlib import Path
import asyncio
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.schema import Document
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import StorageContext
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.base.llms.base import BaseLLM
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
import chromadb
from abc import ABC, abstractmethod

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
        pass
    
    async def parse(self, file_path: str) -> List[Document]:
        """Parse documents from directory path"""
        try:
            # Convert to Path object for better handling
            path = Path(file_path)
            
            # Validate path exists
            if not path.exists():
                raise FileNotFoundError(f"Path {file_path} does not exist")
            
            # Use SimpleDirectoryReader to load documents
            reader = SimpleDirectoryReader(
                input_dir=str(path) if path.is_dir() else str(path.parent),
                filename_as_id=True
            )
            documents = reader.load_data()
            
            return documents
        except Exception as e:
            raise Exception(f"Failed to parse documents: {str(e)}")

class VectorStore:
    """Service responsible for storing and retrieving embeddings from ChromaDB"""
    
    def __init__(self, persist_path: str = "./chroma_db", collection_name: str = "rag_collection"):
        self.persist_path = persist_path
        self.collection_name = collection_name
        self.client = chromadb.PersistentClient(path=persist_path)
        self.chroma_collection = self.client.get_or_create_collection(collection_name)
        self.vector_store = ChromaVectorStore(chroma_collection=self.chroma_collection)
    
    async def store(self, documents: List[Document]) -> ProcessResult:
        """Store documents with embeddings in vector database"""
        try:
            # Create storage context
            storage_context = StorageContext.from_defaults(vector_store=self.vector_store)
            
            # Create index which will automatically store in ChromaDB
            index = VectorStoreIndex.from_documents(
                documents, 
                storage_context=storage_context
            )
            
            return ProcessResult(
                success=True,
                document_count=len(documents),
                chunk_count=len(documents),  # Simplified - in practice this would be actual chunk count
                metadata={
                    "collection_name": self.collection_name,
                    "storage_path": self.persist_path
                }
            )
        except Exception as e:
            raise Exception(f"Failed to store embeddings: {str(e)}")
    
    def get_index(self) -> VectorStoreIndex:
        """Retrieve existing index from vector store"""
        try:
            storage_context = StorageContext.from_defaults(vector_store=self.vector_store)
            # This assumes documents were already stored
            index = VectorStoreIndex.from_vector_store(
                vector_store=self.vector_store,
                storage_context=storage_context
            )
            return index
        except Exception as e:
            raise Exception(f"Failed to retrieve index: {str(e)}")

class RetrievalService:
    """Service responsible for retrieving relevant documents based on queries"""
    
    def __init__(self, vector_store: VectorStore):
        self.vector_store = vector_store
    
    async def search(self, query: str, top_k: int = 5) -> QueryResponse:
        """Search for relevant documents based on query"""
        try:
            # Get existing index
            index = self.vector_store.get_index()
            
            # Create query engine
            query_engine = index.as_query_engine(similarity_top_k=top_k)
            
            # Execute query
            response = query_engine.query(query)
            
            # Extract source documents
            source_nodes = getattr(response, 'source_nodes', [])
            source_documents = [node.node for node in source_nodes] if source_nodes else []
            
            return QueryResponse(
                content=str(response),
                source_documents=source_documents,
                metadata={
                    "query": query,
                    "top_k": top_k,
                    "similarity_scores": [node.score for node in source_nodes] if source_nodes else []
                }
            )
        except Exception as e:
            raise Exception(f"Failed to search documents: {str(e)}")

class LLMOrchestrator:
    """Service responsible for generating responses using LLM"""
    
    def __init__(self, llm: Optional[BaseLLM] = None):
        self.llm = llm or OpenAI()
    
    async def generate(self, query: str, context: List[Document]) -> QueryResponse:
        """Generate response using LLM with provided context"""
        try:
            # Format context for prompt
            context_text = "\n\n".join([doc.text for doc in context[:3]])  # Limit context
            
            # Create prompt with context
            prompt = f"""
            Context information is below.
            ---------------------
            {context_text}
            ---------------------
            Given the context information and not prior knowledge, 
            answer the query.
            Query: {query}
            Answer:
            """
            
            # Generate response
            response = self.llm.complete(prompt)
            
            return QueryResponse(
                content=str(response),
                source_documents=context,
                metadata={
                    "model": self.llm.__class__.__name__,
                    "prompt_length": len(prompt)
                }
            )
        except Exception as e:
            raise Exception(f"Failed to generate response: {str(e)}")


# # Usage Example
# async def main():
#     """Example of how to use the services together"""
    
#     # Initialize services
#     parser = DocumentParser()
#     embedding_service = EmbeddingService()
#     vector_store = VectorStore("./chroma_db", "my_documents")
#     retrieval_service = RetrievalService(vector_store)
#     llm_orchestrator = LLMOrchestrator()
    
#     # Initialize orchestrator
#     workflow = RAGWorkflowOrchestrator(
#         parser, embedding_service, vector_store, retrieval_service, llm_orchestrator
#     )
    
#     # Example usage
#     try:
#         # Process documents
#         result = await workflow.process_document("./documents/")
#         print(f"Processed {result.document_count} documents")
        
#         # Query documents
#         response = await workflow.query_document("What is the main topic?")
#         print(f"Response: {response.content}")
        
#     except Exception as e:
#         print(f"Error: {e}")

# if __name__ == "__main__":
#     # Run the async main function
#     asyncio.run(main())
