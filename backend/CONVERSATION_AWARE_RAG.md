# Conversation-Aware RAG Implementation

This document describes the conversation-aware RAG (Retrieval-Augmented Generation) functionality that has been added to the document analyzer application.

## 🌟 Features

### 1. Persistent Chat History
- **SQLite Database Storage**: All conversations are stored in a SQLite database (`chat_history.db`)
- **Session-based Organization**: Conversations are organized by `session_id` for multi-user support
- **Automatic Persistence**: Every user query and assistant response is automatically saved

### 2. Context-Aware Responses
- **Memory Integration**: The system remembers previous conversations within a session
- **Contextual Understanding**: Follow-up questions can reference previous parts of the conversation
- **Token Limit Management**: Chat history is managed with a 3000 token limit to stay within API constraints

### 3. Dual Query Modes
- **With History**: `query_document_with_history()` - Uses conversation context
- **Without History**: `query_document()` - Traditional stateless querying
- **User Choice**: API endpoint allows users to enable/disable history via `use_history` parameter

## 🏗️ Architecture

### Core Components

#### 1. Database Layer (`database.py`)
```python
class ChatMessage(Base):
    id = Column(Integer, primary_key=True)
    session_id = Column(String, index=True)
    role = Column(String)  # 'user' or 'assistant'
    content = Column(String)
    timestamp = Column(DateTime, default=datetime.utcnow)

class DatabaseChatHistory:
    def add_message(self, session_id: str, role: str, content: str)
    def get_history(self, session_id: str, limit: int = 10)
```

#### 2. Chat Engine (`chat_engine.py`)
```python
class ChatHistoryManager:
    def create_chat_engine_with_history(self, session_id: str)
    def save_chat_interaction(self, session_id: str, user_input: str, response: str)
```

#### 3. Workflow Integration (`workflow.py`)
```python
class RAGWorkflowOrchestrator:
    async def query_document_with_history(self, query: str, session_id: str)
    # Existing method still available:
    async def query_document(self, query: str)
```

## 🔌 API Endpoints

### Enhanced Answer Endpoint
```http
GET /answer?query={query}&session_id={session_id}&use_history={true|false}
```

**Parameters:**
- `query` (required): The user's question
- `session_id` (optional): Session identifier (defaults to "default")
- `use_history` (optional): Whether to use conversation history (defaults to true)
- `top_k` (optional): Number of documents to retrieve (defaults to 5)

**Response:**
```json
{
  "response": "Assistant's response text",
  "session_id": "session_identifier",
  "used_history": true
}
```

### Conversation Management Endpoints

#### Get Conversation History
```http
GET /conversation/{session_id}?limit={limit}
```

**Response:**
```json
{
  "session_id": "session_identifier",
  "history": [
    {"role": "user", "content": "What is this document about?"},
    {"role": "assistant", "content": "This document is about..."}
  ],
  "message_count": 2
}
```

#### Clear Conversation History
```http
DELETE /conversation/{session_id}
```

**Response:**
```json
{
  "session_id": "session_identifier",
  "cleared_messages": 4,
  "message": "Successfully cleared conversation history"
}
```

## 🚀 Usage Examples

### 1. Basic Conversation Flow
```python
# First query
response1 = await workflow.query_document_with_history(
    "What is the main topic of this document?", 
    session_id="user123"
)

# Follow-up query (remembers context)
response2 = await workflow.query_document_with_history(
    "Can you elaborate on that?", 
    session_id="user123"
)

# Context-dependent query
response3 = await workflow.query_document_with_history(
    "What are the implications?", 
    session_id="user123"
)
```

### 2. API Usage
```bash
# Start a conversation
curl "http://localhost:8000/answer?query=What is this document about?&session_id=user123"

# Follow-up (remembers previous context)
curl "http://localhost:8000/answer?query=Can you give me more details?&session_id=user123"

# View conversation history
curl "http://localhost:8000/conversation/user123"

# Clear conversation
curl -X DELETE "http://localhost:8000/conversation/user123"
```

### 3. Testing
Run the test script to see the system in action:
```bash
cd backend
python test_conversation_rag.py
```

## 🔧 Configuration

### Database Configuration
The SQLite database is automatically created in the backend directory as `chat_history.db`. You can modify the database URL in `DatabaseChatHistory.__init__()`:

```python
def __init__(self, database_url: str = "sqlite:///chat_history.db"):
```

### Memory Configuration
The chat memory is configured with a 3000 token limit. You can adjust this in `ChatHistoryManager.create_chat_engine_with_history()`:

```python
memory = ChatMemoryBuffer.from_defaults(
    chat_history=chat_messages,
    token_limit=3000  # Adjust as needed
)
```

## 🎯 Benefits

1. **Enhanced User Experience**: Users can have natural, flowing conversations
2. **Context Preservation**: No need to repeat information in follow-up questions
3. **Multi-User Support**: Each session maintains its own conversation history
4. **Flexible Usage**: Can be enabled/disabled per request
5. **Persistent Storage**: Conversations survive application restarts
6. **API Backward Compatibility**: Existing clients continue to work without changes

## 🔮 Future Enhancements

- **Conversation Summarization**: Automatically summarize long conversations
- **Semantic Search**: Search across conversation history
- **Export Functionality**: Export conversations to different formats
- **Advanced Memory Management**: Intelligent context window management
- **User Authentication**: Link conversations to authenticated users
- **Conversation Analytics**: Track conversation patterns and insights

## 🧪 Testing

The system includes a comprehensive test script (`test_conversation_rag.py`) that demonstrates:
- Conversation flow with context preservation
- History retrieval and management
- Comparison between context-aware and traditional responses

Run the test to verify the implementation works correctly in your environment.
