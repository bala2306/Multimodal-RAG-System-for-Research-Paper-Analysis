import { useState } from 'react'
import { uploadAdvanced } from '../services/api'

function AdvancedIngestion() {
    const [file, setFile] = useState(null)
    const [dragging, setDragging] = useState(false)
    const [loading, setLoading] = useState(false)
    const [progress, setProgress] = useState(0)
    const [result, setResult] = useState(null)
    const [error, setError] = useState(null)

    const handleDragOver = (e) => {
        e.preventDefault()
        setDragging(true)
    }

    const handleDragLeave = () => {
        setDragging(false)
    }

    const handleDrop = (e) => {
        e.preventDefault()
        setDragging(false)

        const droppedFile = e.dataTransfer.files[0]
        if (droppedFile && droppedFile.type === 'application/pdf') {
            setFile(droppedFile)
            setError(null)
        } else {
            setError('Please drop a PDF file')
        }
    }

    const handleFileChange = (e) => {
        const selectedFile = e.target.files[0]
        if (selectedFile) {
            setFile(selectedFile)
            setError(null)
        }
    }

    const handleUpload = async () => {
        if (!file) {
            setError('Please select a file')
            return
        }

        setLoading(true)
        setError(null)
        setResult(null)
        setProgress(0)

        try {
            const data = await uploadAdvanced(file, setProgress)
            setResult(data)
            setFile(null)
        } catch (err) {
            setError(err.response?.data?.detail || err.message || 'Failed to upload file')
        } finally {
            setLoading(false)
            setProgress(0)
        }
    }

    return (
        <div className="card">
            <h2>🚀 Advanced Ingestion</h2>
            <p style={{ color: 'var(--text-secondary)', marginBottom: '2rem' }}>
                Multimodal processing with table and image extraction
            </p>

            <div
                className={`upload-zone ${dragging ? 'dragging' : ''}`}
                onDragOver={handleDragOver}
                onDragLeave={handleDragLeave}
                onDrop={handleDrop}
                onClick={() => document.getElementById('file-input-advanced').click()}
            >
                <svg className="upload-icon-svg" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                    <path d="M21.44 11.05l-9.19 9.19a6 6 0 01-8.49-8.49l9.19-9.19a4 4 0 015.66 5.66l-9.2 9.19a2 2 0 01-2.83-2.83l8.49-8.48"
                        stroke="url(#gradient)"
                        strokeWidth="2"
                        strokeLinecap="round"
                        strokeLinejoin="round" />
                    <defs>
                        <linearGradient id="gradient" x1="0%" y1="0%" x2="100%" y2="100%">
                            <stop offset="0%" stopColor="#667eea" />
                            <stop offset="100%" stopColor="#764ba2" />
                        </linearGradient>
                    </defs>
                </svg>
                <p>
                    {file ? (
                        <strong>{file.name}</strong>
                    ) : (
                        <>Drag & drop your PDF here, or click to browse</>
                    )}
                </p>
                <small>PDF files with tables and images supported</small>
            </div>

            <input
                id="file-input-advanced"
                type="file"
                accept=".pdf"
                onChange={handleFileChange}
                style={{ display: 'none' }}
            />

            {file && (
                <div className="mt-3">
                    <button
                        className="btn btn-primary"
                        onClick={handleUpload}
                        disabled={loading}
                    >
                        {loading ? (
                            <>
                                <span className="loading"></span>
                                <span style={{ marginLeft: '0.5rem' }}>
                                    Processing... {progress}%
                                </span>
                            </>
                        ) : (
                            'Upload & Process'
                        )}
                    </button>
                </div>
            )}

            {error && (
                <div className="alert alert-error mt-3">
                    <strong>Error:</strong> {error}
                </div>
            )}

            {result && (
                <div className="alert alert-success mt-3">
                    <h3 style={{ marginBottom: '1rem' }}>✅ Advanced Processing Complete!</h3>

                    <div className="result-stats">
                        <div className="stat-card">
                            <div className="stat-label">Document ID</div>
                            <div className="stat-value" style={{ fontSize: '0.9rem', wordBreak: 'break-all' }}>
                                {result.document_id}
                            </div>
                        </div>

                        <div className="stat-card">
                            <div className="stat-label">Filename</div>
                            <div className="stat-value" style={{ fontSize: '1rem' }}>
                                {result.filename}
                            </div>
                        </div>

                        <div className="stat-card">
                            <div className="stat-label">Total Pages</div>
                            <div className="stat-value">{result.total_pages}</div>
                        </div>

                        <div className="stat-card">
                            <div className="stat-label">Text Chunks</div>
                            <div className="stat-value">{result.chunks_created}</div>
                        </div>

                        <div className="stat-card">
                            <div className="stat-label">Tables Extracted</div>
                            <div className="stat-value">{result.tables_extracted}</div>
                        </div>

                        <div className="stat-card">
                            <div className="stat-label">Images Extracted</div>
                            <div className="stat-value">{result.images_extracted}</div>
                        </div>

                        <div className="stat-card">
                            <div className="stat-label">Visual Elements</div>
                            <div className="stat-value">{result.visual_elements_count}</div>
                        </div>
                    </div>

                    <div className="alert alert-info" style={{ marginTop: '1.5rem' }}>
                        <strong>📊 Multimodal Processing:</strong> Extracted {result.tables_extracted} tables
                        and {result.images_extracted} images with semantic descriptions
                    </div>

                    <p style={{ marginTop: '1rem', color: 'var(--text-secondary)' }}>
                        {result.message}
                    </p>
                </div>
            )}
        </div>
    )
}

export default AdvancedIngestion
