import os
import logging
from typing import List

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.chat_history import BaseChatMessageHistory

from app.core.database import DatabaseChatHistory
from app.core.service import VectorStore, QueryResponse
from app.services.prompt_manager import get_model_config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DatabaseBackedChatHistory(BaseChatMessageHistory):
    """Chat message history backed by database storage."""
    
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.db_history = DatabaseChatHistory()
        
    @property
    def messages(self) -> List[BaseMessage]:
        """Retrieve messages from database and convert to LangChain format."""
        history_data = self.db_history.get_history(self.session_id)
        messages = []
        for msg in history_data:
            if msg["role"] == "user":
                messages.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                messages.append(AIMessage(content=msg["content"]))
        return messages
    
    def add_user_message(self, message: str) -> None:
        """Add a user message to the database."""
        self.db_history.add_message(self.session_id, "user", message)
    
    def add_ai_message(self, message: str) -> None:
        """Add an AI message to the database."""
        self.db_history.add_message(self.session_id, "assistant", message)
    
    def clear(self) -> None:
        """Clear all messages for this session."""
        self.db_history.clear_session_history(self.session_id)


class LangChainChatHistoryManager:
    """LangChain-based chat history manager with conversational retrieval."""
    
    def __init__(self, vector_store: VectorStore):
        api_key = os.environ.get("OPENAI_API_KEY")
        model_config = get_model_config()

        self.llm = ChatOpenAI(
            api_key=api_key,
            base_url="https://api.deepinfra.com/v1/openai",
            model=model_config.get('model_name', 'Qwen/Qwen2.5-72B-Instruct'),
            temperature=model_config.get('temperature', 0.7),
        )

        self.vector_store = vector_store
        self.retriever = vector_store.get_retriever(k=5)

        self.contextualize_q_prompt = ChatPromptTemplate.from_messages([
            ("system", """Given a chat history and the latest user question which might reference context in the chat history, 
            formulate a standalone question which can be understood without the chat history. 
            Do NOT answer the question, just reformulate it if needed and otherwise return it as is."""),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])

        self.qa_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. 
            If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.
            
            Context: {context}"""),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])
        
        self.history_aware_retriever = create_history_aware_retriever(
            self.llm, self.retriever, self.contextualize_q_prompt
        )
        
        self.question_answer_chain = create_stuff_documents_chain(
            self.llm, self.qa_prompt
        )
        
        self.rag_chain = create_retrieval_chain(
            self.history_aware_retriever, self.question_answer_chain
        )
    
    def get_session_history(self, session_id: str) -> BaseChatMessageHistory:
        """Get chat history for a session."""
        return DatabaseBackedChatHistory(session_id)
    
    async def query_with_history(self, query: str, session_id: str = "default") -> QueryResponse:
        """Query with conversation history awareness using LangChain."""
        try:
            logger.info(f"Processing query with history: {query[:50]}... (session: {session_id})")
            
            chat_history = self.get_session_history(session_id)
            
            result = await self.rag_chain.ainvoke({
                "input": query,
                "chat_history": chat_history.messages
            })
            
            chat_history.add_user_message(query)
            chat_history.add_ai_message(result["answer"])
            
            response = QueryResponse(
                content=result["answer"],
                source_documents=result.get("context", []),
                metadata={
                    "session_id": session_id,
                    "has_history": True,
                    "num_source_docs": len(result.get("context", []))
                }
            )
            
            logger.info("Successfully processed query with history")
            return response
            
        except Exception as e:
            logger.error(f"Failed to process query with history: {str(e)}", exc_info=True)
            raise Exception(f"Failed to process query with history: {str(e)}")
    
