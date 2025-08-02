import { useState, useRef } from 'react'
import './ChatPanel.css'

const ChatPanel = ({ messages, onSendMessage, isLoading, hasDocuments }) => {
  const [message, setMessage] = useState('')
  const inputRef = useRef()

  const handleChange = (e) => {
    setMessage(e.target.value)
  }

  const handleSubmit = (e) => {
    e.preventDefault()
    if (message.trim()) {
      onSendMessage(message)
      setMessage('')
      inputRef.current.focus()
    }
  }

  return (
    <div className="chat-panel">
      <div className="chat-header">
        <h2>Chat</h2>
      </div>

      <div className="chat-messages">
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
          placeholder={hasDocuments ? "Type your message here..." : "Upload documents to start chatting..."}
          value={message}
          onChange={handleChange}
          disabled={!hasDocuments}
          ref={inputRef}
        />
        <button type="submit" disabled={!hasDocuments}>Send</button>
      </form>
    </div>
  )
}

export default ChatPanel

