import { useState } from 'react'
import './App.css'
import SourcesPanel from './components/SourcesPanel'
import ChatPanel from './components/ChatPanel'

function App() {
  const [sources, setSources] = useState([])
  const [messages, setMessages] = useState([])
  const [isLoading, setIsLoading] = useState(false)

  const handleFileUpload = (files) => {
    const newSources = files.map(file => ({
      id: Date.now() + Math.random(),
      name: file.name,
      type: file.type,
      size: file.size,
      file: file,
      uploaded: true
    }))
    setSources(prev => [...prev, ...newSources])
  }

  const handleRemoveSource = (id) => {
    setSources(prev => prev.filter(source => source.id !== id))
  }

  const handleSendMessage = async (messageText) => {
    if (!messageText || !messageText.trim()) return

    const userMessage = {
      id: Date.now(),
      type: 'user',
      content: messageText.trim(),
      timestamp: new Date()
    }

    setMessages(prev => [...prev, userMessage])
    setIsLoading(true)

    try {
      // Send message to API
      const url = new URL('http://127.0.0.1:8000/answer')
      url.searchParams.append('query', messageText.trim())
      
      const response = await fetch(url, {
        method: 'GET',
      })

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`)
      }

      const data = await response.json()
      console.log('API Response:', data)
      
      // Add AI response to chat
      const aiMessage = {
        id: Date.now() + 1,
        type: 'ai',
        content: data.answer || data.response || "Sorry, I couldn't process that.",
        timestamp: new Date()
      }
      setMessages(prev => [...prev, aiMessage])
    } catch (error) {
      console.error('Error sending message:', error)
      // Add error message to chat
      const errorMessage = {
        id: Date.now() + 1,
        type: 'error',
        content: "Failed to get response. Please try again.",
        timestamp: new Date()
      }
      setMessages(prev => [...prev, errorMessage])
    } finally {
      setIsLoading(false)
    }
  }

  return (
    <div className="app">
      <header className="app-header">
        <div className="header-content">
          <div className="app-icon">📝</div>
          <h1>Document Analyzer</h1>
        </div>
      </header>
      
      <main className="app-main">
        <SourcesPanel 
          sources={sources}
          onFileUpload={handleFileUpload}
          onRemoveSource={handleRemoveSource}
        />
        <ChatPanel 
          messages={messages}
          onSendMessage={handleSendMessage}
          isLoading={isLoading}
          hasDocuments={sources.length > 0}
        />
      </main>
    </div>
  )
}

export default App
