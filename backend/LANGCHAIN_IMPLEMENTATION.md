# LangChain-based RAG Implementation

This document describes the complete LangChain implementation that replaces the previous LlamaIndex-based system. The new implementation uses modern LangChain components for conversation-aware retrieval-augmented generation (RAG).

## 🏗️ Architecture Overview

The LangChain implementation consists of several key components:

### Core Components

1. **LangChainChatHistoryManager** (`app/core/langchain_chat_engine.py`)
   - Manages conversation history using LangChain's chat memory
   - Implements history-aware retrieval chains
   - Provides database-backed chat history storage

2. **LangChainRAGWorkflowOrchestrator** (`app/core/langchain_workflow.py`)
   - Main orchestrator for the RAG pipeline
   - Coordinates document processing and querying
   - Supports both conversational and simple queries

3. **DatabaseBackedChatHistory** (`app/core/langchain_chat_engine.py`)
   - Custom implementation of `BaseChatMessageHistory`
   - Stores chat messages in SQLite database
   - Provides seamless integration with LangChain's memory system

## 🔧 Key LangChain Components Used

### 1. History-Aware Retrieval
```python
from langchain.chains import create_history_aware_retriever

# Creates a retriever that reformulates queries based on chat history
history_aware_retriever = create_history_aware_retriever(
    llm, retriever, contextualize_prompt
)
```

### 2. Document Chain
```python
from langchain.chains.combine_documents import create_stuff_documents_chain

# Creates a chain that combines documents with LLM for answer generation
question_answer_chain = create_stuff_documents_chain(
    llm, qa_prompt
)
```

### 3. Retrieval Chain
```python
from langchain.chains import create_retrieval_chain

# Combines history-aware retrieval with document processing
rag_chain = create_retrieval_chain(
    history_aware_retriever, question_answer_chain
)
```

### 4. Chat Message History
```python
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import HumanMessage, AIMessage

# Custom database-backed chat history implementation
class DatabaseBackedChatHistory(BaseChatMessageHistory):
    # Stores messages in SQLite database
    # Integrates with LangChain's memory system
```

### 5. Prompt Templates
```python
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# Context-aware prompt for query reformulation
contextualize_q_prompt = ChatPromptTemplate.from_messages([
    ("system", "..."),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

# Answer generation prompt with context and history
qa_prompt = ChatPromptTemplate.from_messages([
    ("system", "..."),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])
```

## 🚀 Features

### 1. Conversation-Aware Retrieval
- **History Context**: Uses previous conversations to understand follow-up questions
- **Query Reformulation**: Automatically reformulates ambiguous queries using chat history
- **Contextual Understanding**: Maintains conversation context across multiple exchanges

### 2. Dual Query Modes
- **With History** (`query_document_with_history`): Uses conversation context
- **Without History** (`query_document`): Traditional stateless querying
- **API Control**: Users can enable/disable history per request

### 3. Database Persistence
- **SQLite Storage**: All conversations stored in `chat_history.db`
- **Session Management**: Conversations organized by session ID
- **Message Tracking**: Both user queries and AI responses are stored

### 4. Flexible Architecture
- **Modular Design**: Easy to swap components
- **Async Support**: Full async/await support throughout
- **Error Handling**: Comprehensive error handling and logging

## 📋 API Endpoints

### Query with History
```http
GET /answer?query={query}&session_id={session_id}&use_history=true
```

**Example:**
```bash
# First query
curl "http://localhost:8000/answer?query=What is LangChain?&session_id=user123"

# Follow-up query (remembers context)
curl "http://localhost:8000/answer?query=What are its key components?&session_id=user123"
```

### Query without History
```http
GET /answer?query={query}&use_history=false
```

### Conversation Management
```http
# Get conversation history
GET /conversation/{session_id}?limit={limit}

# Clear conversation history
DELETE /conversation/{session_id}
```

## 🔄 Migration from LlamaIndex

The new implementation replaces the LlamaIndex components as follows:

| LlamaIndex Component | LangChain Replacement |
|---------------------|----------------------|
| `ChatEngine` | `create_retrieval_chain` + `create_history_aware_retriever` |
| `ChatMemoryBuffer` | `DatabaseBackedChatHistory` (custom `BaseChatMessageHistory`) |
| `VectorStoreIndex.as_chat_engine()` | `create_stuff_documents_chain` + retriever |
| `ChatMessage` | `HumanMessage` + `AIMessage` |
| LlamaIndex prompts | `ChatPromptTemplate` with `MessagesPlaceholder` |

## 🧪 Testing

### Running Tests
```bash
cd backend
python test_langchain_rag.py
```

### Test Coverage
The test script covers:
- Document processing and storage
- Conversational queries with context
- Follow-up questions that reference previous context
- Simple queries without history
- Conversation history retrieval

## 📁 File Structure

```
app/core/
├── langchain_chat_engine.py     # LangChain chat history manager
├── langchain_workflow.py        # LangChain workflow orchestrator
├── database.py                  # Database models and chat history
├── service.py                   # Document processing and vector store
└── orchestrator.py              # LLM orchestrator

app/api/routes/
└── upload.py                    # Updated API routes using LangChain

test_langchain_rag.py            # Comprehensive test script
```

## 🎯 Benefits of LangChain Implementation

1. **Better Conversation Memory**: More sophisticated history handling
2. **Improved Context Understanding**: Better query reformulation and context preservation
3. **Modular Architecture**: Easier to extend and maintain
4. **Industry Standard**: Uses widely adopted LangChain framework
5. **Better Documentation**: Extensive community and official documentation
6. **Future-Proof**: Regular updates and active development

## 🔧 Configuration

### Environment Variables
```bash
export OPENAI_API_KEY="your_api_key"
```

### Model Configuration
Update `app/services/prompt_manager.py` to configure:
- Model name
- Temperature
- Max tokens
- API base URL

### Database Configuration
The SQLite database (`chat_history.db`) is automatically created. For production, consider:
- PostgreSQL or MySQL for better performance
- Connection pooling
- Database migrations

## 🚀 Deployment

1. **Install Dependencies**:
   ```bash
   pip install langchain langchain-chroma langchain-core langchain-text-splitters langchain-openai langchain-community
   ```

2. **Set Environment Variables**:
   ```bash
   export OPENAI_API_KEY="your_key"
   ```

3. **Start the Application**:
   ```bash
   uvicorn app.main:app --reload
   ```

4. **Run Tests**:
   ```bash
   python test_langchain_rag.py
   ```

## 🔮 Future Enhancements

- **Streaming Responses**: Add streaming support for real-time responses
- **Advanced Memory Management**: Implement conversation summarization
- **Multi-Modal Support**: Add support for images and other media
- **Custom Retrievers**: Implement domain-specific retrieval strategies
- **Conversation Analytics**: Add conversation insights and analytics

## 📝 Notes

- The implementation maintains backward compatibility with existing API endpoints
- Database schema remains unchanged for seamless migration
- All LlamaIndex dependencies can be removed after testing
- The system supports both development and production environments
