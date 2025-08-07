#!/usr/bin/env python3
"""
Test script to verify the conversation-aware RAG system
This script demonstrates how the system can maintain context across multiple interactions
"""

import asyncio
import sys
import os

# Add the app directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

from app.core.service import DocumentParser, VectorStore, RetrievalService
from app.core.orchestrator import LLMOrchestrator
from app.core.workflow import RAGWorkflowOrchestrator
from app.core.database import DatabaseChatHistory


async def test_conversation_aware_rag():
    """Test the conversation-aware RAG system"""
    print("Starting Conversation-Aware RAG Test\n")
    
    # Initialize components
    print("Initializing RAG components...")
    parser = DocumentParser()
    vector_store = VectorStore("./chroma_db", "test_documents")
    retrieval_service = RetrievalService(vector_store)
    llm_orchestrator = LLMOrchestrator()
    
    # Create workflow orchestrator
    workflow = RAGWorkflowOrchestrator(
        parser, vector_store, retrieval_service, llm_orchestrator
    )
    
    # Test session ID
    test_session = "test_session_123"
    
    # Clear any existing conversation history for clean test
    print(f"🧹 Clearing existing conversation history for session: {test_session}")
    db_history = DatabaseChatHistory()
    db = db_history.SessionLocal()
    try:
        from app.core.database import ChatMessage
        db.query(ChatMessage).filter(ChatMessage.session_id == test_session).delete()
        db.commit()
        print("History cleared")
    finally:
        db.close()
    
    # Process documents first (if uploads directory exists)
    uploads_path = "./uploads/"
    if os.path.exists(uploads_path) and os.listdir(uploads_path):
        print(f"Processing documents from {uploads_path}...")
        try:
            result = await workflow.process_document(uploads_path)
            print(f"Processed {result.document_count} documents")
        except Exception as e:
            print(f"Could not process documents: {e}")
    else:
        print("No uploads directory found or it's empty. Conversation will work but without document context.")
    
    # Test conversation flow
    print("\nTesting conversation flow:\n")
    
    # First interaction
    print("User: What is this document about?")
    try:
        response1 = await workflow.query_document_with_history(
            "What is this document about?", 
            test_session
        )
        print(f"🤖 Assistant: {response1.content}\n")
    except Exception as e:
        print(f"Error in first query: {e}\n")
        return
    
    # Second interaction (should remember context)
    print("👤 User: Can you elaborate on the main points?")
    try:
        response2 = await workflow.query_document_with_history(
            "Can you elaborate on the main points?", 
            test_session
        )
        print(f"Assistant: {response2.content}\n")
    except Exception as e:
        print(f"Error in second query: {e}\n")
        return
    
    # Third interaction (context-dependent follow-up)
    print("👤 User: What did you mean by that?")
    try:
        response3 = await workflow.query_document_with_history(
            "What did you mean by that?", 
            test_session
        )
        print(f"Assistant: {response3.content}\n")
    except Exception as e:
        print(f"Error in third query: {e}\n")
        return
    
    # Check conversation history
    print("📜 Retrieving conversation history:")
    history = db_history.get_history(test_session, limit=10)
    for i, msg in enumerate(history, 1):
        role = "👤" if msg["role"] == "user" else "🤖"
        print(f"{i}. {role} {msg['role'].title()}: {msg['content'][:100]}...")
    
    print(f"\nTest completed! Total messages in history: {len(history)}")
    
    # Test without history (for comparison)
    print("\nTesting same query without conversation history:")
    try:
        response_no_history = await workflow.query_document(
            "What did you mean by that?"
        )
        print(f"Assistant (no history): {response_no_history.content}\n")
        print("Notice how the response differs without conversation context!")
    except Exception as e:
        print(f"Error in no-history query: {e}")


if __name__ == "__main__":
    # Run the test
    asyncio.run(test_conversation_aware_rag())
