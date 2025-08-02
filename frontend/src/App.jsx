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

  const handleSendMessage = async (message) => {
    if (!message.trim()) return

    const userMessage = {
      id: Date.now(),
      type: 'user',
      content: message,
      timestamp: new Date()
    }

    setMessages(prev => [...prev, userMessage])
    setIsLoading(true)

    // Simulate AI response (replace with actual API call)
    setTimeout(() => {
      const aiMessage = {
        id: Date.now() + 1,
        type: 'ai',
        content: `I understand your question: "${message}". Based on the uploaded documents, I can help you analyze the content. Please note that this is a demo response - in a real implementation, this would process your documents using RAG.`,
        timestamp: new Date()
      }
      setMessages(prev => [...prev, aiMessage])
      setIsLoading(false)
    }, 1500)
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
