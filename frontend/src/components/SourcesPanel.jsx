import { useState } from 'react'
import { useDropzone } from 'react-dropzone'
import './SourcesPanel.css'

const SourcesPanel = ({ sources, onFileUpload, onRemoveSource }) => {
  const [isExpanded, setIsExpanded] = useState(false)

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop: onFileUpload,
    accept: {
      'application/pdf': ['.pdf'],
      'text/plain': ['.txt'],
      'application/msword': ['.doc'],
      'application/vnd.openxmlformats-officedocument.wordprocessingml.document': ['.docx'],
      'text/markdown': ['.md']
    },
    multiple: true
  })

  const formatFileSize = (bytes) => {
    if (bytes === 0) return '0 Bytes'
    const k = 1024
    const sizes = ['Bytes', 'KB', 'MB', 'GB']
    const i = Math.floor(Math.log(bytes) / Math.log(k))
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i]
  }

  const getFileIcon = (type) => {
    if (type.includes('pdf')) return '📄'
    if (type.includes('word') || type.includes('document')) return '📝'
    if (type.includes('text')) return '📃'
    return '📄'
  }

  return (
    <div className={`sources-panel ${isExpanded ? 'expanded' : ''}`}>
      <div className="sources-header">
        <h2>Sources</h2>
        <button 
          className="expand-button"
          onClick={() => setIsExpanded(!isExpanded)}
        >
          {isExpanded ? '◂' : '▸'}
        </button>
      </div>

      <div className="sources-content">
        <div className="upload-section">
        </div>

        <div 
          {...getRootProps()} 
          className={`dropzone ${isDragActive ? 'active' : ''}`}
        >
          <input {...getInputProps()} />
          <div className="dropzone-content">
            <div className="dropzone-icon">📄</div>
            <h3>Add a source to get started</h3>
            <p>
              Click Add source above to add PDFs, text.
            </p>
            <button className="upload-button">Upload a source</button>
          </div>
        </div>

        {sources.length > 0 && (
          <div className="sources-list">
            <h4>Uploaded Sources ({sources.length})</h4>
            {sources.map((source) => (
              <div key={source.id} className="source-item">
                <div className="source-info">
                  <span className="source-icon">{getFileIcon(source.type)}</span>
                  <div className="source-details">
                    <div className="source-name">{source.name}</div>
                    <div className="source-meta">{formatFileSize(source.size)}</div>
                  </div>
                </div>
                <button
                  className="remove-button"
                  onClick={() => onRemoveSource(source.id)}
                  title="Remove source"
                >
                  ×
                </button>
              </div>
            ))}
          </div>
        )}

        <div className="sources-info">
          <p>Saved sources will appear here</p>
        </div>
      </div>
    </div>
  )
}

export default SourcesPanel
