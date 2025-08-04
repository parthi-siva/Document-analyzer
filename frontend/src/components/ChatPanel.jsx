import { useState, useRef } from 'react'
import './ChatPanel.css'

const ChatPanel = ({ messages, onSendMessage, isLoading, hasDocuments }) => {
  const [message, setMessage] = useState('')
  const inputRef = useRef()

  const handleChange = (e) => {
    setMessage(e.target.value)
  }

  const handleSubmit = async (e) => {
    e.preventDefault()
    
    // Validate and trim message
    const trimmedMessage = typeof message === 'string' ? message.trim() : ''
    if (!trimmedMessage || !hasDocuments) return

    // Clear input immediately
    setMessage('')

    // Send message to parent component (App.jsx) which handles the API call
    onSendMessage(trimmedMessage)
  }

  return (
    <div className="chat-panel">
      <div className="chat-header">
        <h2>Chat</h2>
      </div>

      <div className="chat-messages">
        {!hasDocuments && messages.length === 0 && (
          <div className="empty-state">
            <div className="empty-icon">💬</div>
            <h3>Ready to chat with your documents</h3>
            <p>Upload some documents in the Sources panel to get started. Once uploaded, you can ask questions about their content.</p>
          </div>
        )}
        
        {messages.map((msg) => (
          <div key={msg.id} className={`chat-message ${msg.type}`}>
            <div className="message-content">{msg.content}</div>
          </div>
        ))}
        {isLoading && <div className="typing-indicator">Agent is typing...</div>}
      </div>

      <form className="chat-form" onSubmit={handleSubmit}>
        <input
          type="text"
          placeholder={hasDocuments ? "Ask me anything about your documents..." : "Upload documents to start chatting..."}
          value={typeof message === 'string' ? message : ''}
          onChange={handleChange}
          disabled={!hasDocuments}
          ref={inputRef}
        />
        <button
          type="submit"
          disabled={!hasDocuments || !message || !message.trim()}
        >
          Send
        </button>
      </form>
    </div>
  )
}

export default ChatPanel
