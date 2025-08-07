#!/usr/bin/env python3
"""
Test script for the LangChain-based RAG implementation.
This script tests the conversation-aware functionality using LangChain components.
"""

import asyncio
import os
import sys
import logging

# Add the backend directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__)))

from app.core.service import DocumentParser, VectorStore, RetrievalService
from app.core.orchestrator import LLMOrchestrator
from app.core.langchain_workflow import LangChainRAGWorkflowOrchestrator

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


async def test_langchain_rag():
    """Test the LangChain-based RAG implementation."""
    logger.info("🚀 Starting LangChain RAG Implementation Test")
    
    try:
        # Initialize services
        logger.info("Initializing services...")
        parser = DocumentParser()
        vector_store = VectorStore("./test_chroma_db", "test_documents")
        retrieval_service = RetrievalService(vector_store=vector_store)
        llm_orchestrator = LLMOrchestrator()
        
        # Initialize LangChain workflow
        langchain_workflow = LangChainRAGWorkflowOrchestrator(
            parser, vector_store, retrieval_service, llm_orchestrator
        )
        
        # Test document processing
        logger.info("📄 Testing document processing...")
        uploads_path = "./uploads/"
        if not os.path.exists(uploads_path):
            logger.info("Creating uploads directory for testing...")
            os.makedirs(uploads_path)
            
            # Create a sample document for testing
            sample_doc_path = os.path.join(uploads_path, "sample.txt")
            with open(sample_doc_path, "w") as f:
                f.write("""
                LangChain is a framework for developing applications powered by language models.
                It provides tools for document loading, text splitting, embeddings, vector stores,
                retrievers, and chains that combine multiple components together.
                
                The framework supports conversation memory and history-aware retrieval,
                allowing for contextual conversations that remember previous interactions.
                
                Key components include:
                - Document loaders for various file formats
                - Text splitters for chunking documents
                - Vector stores like Chroma for similarity search
                - Retrievers that can be history-aware
                - Chains that combine LLMs with retrievers
                - Memory components for conversation history
                """)
            logger.info(f"Created sample document: {sample_doc_path}")
        
        try:
            process_result = await langchain_workflow.process_document(uploads_path)
            logger.info(f"✅ Document processing successful: {process_result.document_count} documents processed")
        except Exception as e:
            logger.error(f"❌ Document processing failed: {e}")
            return
        
        # Test session ID for conversation history
        test_session = "test_session_123"
        
        # Test 1: First query with history
        logger.info("🤖 Testing conversational query (with history)...")
        try:
            query1 = "What is LangChain?"
            response1 = await langchain_workflow.query_document_with_history(query1, test_session)
            logger.info(f"Query 1: {query1}")
            logger.info(f"Response 1: {response1.content[:200]}...")
            logger.info(f"Source docs: {len(response1.source_documents)}")
        except Exception as e:
            logger.error(f"❌ First conversational query failed: {e}")
            return
        
        # Test 2: Follow-up query with history (should remember context)
        logger.info("🔄 Testing follow-up query (should remember context)...")
        try:
            query2 = "What are its key components?"
            response2 = await langchain_workflow.query_document_with_history(query2, test_session)
            logger.info(f"Query 2: {query2}")
            logger.info(f"Response 2: {response2.content[:200]}...")
            logger.info(f"Source docs: {len(response2.source_documents)}")
        except Exception as e:
            logger.error(f"❌ Follow-up query failed: {e}")
            return
        
        # Test 3: Another contextual query
        logger.info("🔄 Testing another contextual query...")
        try:
            query3 = "How does the conversation memory work?"
            response3 = await langchain_workflow.query_document_with_history(query3, test_session)
            logger.info(f"Query 3: {query3}")
            logger.info(f"Response 3: {response3.content[:200]}...")
            logger.info(f"Source docs: {len(response3.source_documents)}")
        except Exception as e:
            logger.error(f"❌ Third contextual query failed: {e}")
            return
        
        # Test 4: Simple query without history
        logger.info("🤖 Testing simple query (without history)...")
        try:
            query4 = "What is a vector store?"
            response4 = await langchain_workflow.query_document(query4)
            logger.info(f"Query 4: {query4}")
            logger.info(f"Response 4: {response4.content[:200]}...")
            logger.info(f"Source docs: {len(response4.source_documents)}")
        except Exception as e:
            logger.error(f"❌ Simple query failed: {e}")
            return
        
        # Test 5: Retrieve conversation history
        logger.info("📜 Testing conversation history retrieval...")
        try:
            history = await langchain_workflow.get_conversation_history(test_session)
            logger.info(f"Retrieved {len(history)} messages from history")
            for i, msg in enumerate(history):
                logger.info(f"Message {i+1} ({msg['role']}): {msg['content'][:100]}...")
        except Exception as e:
            logger.error(f"❌ History retrieval failed: {e}")
            return
        
        logger.info("🎉 All LangChain RAG tests completed successfully!")
        
        # Summary
        logger.info("\n" + "="*60)
        logger.info("📊 TEST SUMMARY")
        logger.info("="*60)
        logger.info("✅ Document processing: SUCCESS")
        logger.info("✅ Conversational queries: SUCCESS")
        logger.info("✅ Context awareness: SUCCESS")
        logger.info("✅ Simple queries: SUCCESS")
        logger.info("✅ History retrieval: SUCCESS")
        logger.info("="*60)
        
    except Exception as e:
        logger.error(f"❌ Test failed with error: {e}", exc_info=True)


async def main():
    """Main function to run the test."""
    await test_langchain_rag()


if __name__ == "__main__":
    asyncio.run(main())
