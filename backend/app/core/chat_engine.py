from llama_index.core.base.llms.types import ChatMessage
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.llms.openai import OpenAI
from llama_index.core.chat_engine.types import ChatMode
import os

from app.core.database import DatabaseChatHistory
from app.core.custom_llm_wrapper import DeepInfraLLM
from app.services.prompt_manager import get_model_config

class ChatHistoryManager:
    def __init__(self, llm=None, vector_store=None):
        api_key = os.environ.get("OPENAI_API_KEY")
        model_config = get_model_config()
        self.llm = DeepInfraLLM(api_key=api_key, model=model_config.get('model_name', 'Qwen/Qwen3-32B'))
        self.vector_store = vector_store
        self.db_history = DatabaseChatHistory()
        
    def create_chat_engine_with_history(self, session_id: str = "default"):
        # Get history from database
        history_data = self.db_history.get_history(session_id)
        
        # Create memory buffer with existing history
        chat_messages = [
            ChatMessage(role=msg["role"], content=msg["content"])
            for msg in history_data
        ]
        memory = ChatMemoryBuffer.from_defaults(
            chat_history=chat_messages,
            token_limit=3000
        )
        
        # Get index from vector store if available
        if self.vector_store:
            index = self.vector_store.get_index()
        else:
            # Fallback to creating index from data directory
            documents = SimpleDirectoryReader("./uploads/").load_data()
            index = VectorStoreIndex.from_documents(documents)
        
        # Create chat engine
        chat_engine = index.as_chat_engine(
            chat_mode=ChatMode.CONTEXT,
            memory=memory,
            llm=self.llm
        )
        return chat_engine, memory
    
    def save_chat_interaction(self, session_id: str, user_input: str, response: str):
        
        # Save to database
        self.db_history.add_message(session_id, "user", user_input)
        self.db_history.add_message(session_id, "assistant", response)

