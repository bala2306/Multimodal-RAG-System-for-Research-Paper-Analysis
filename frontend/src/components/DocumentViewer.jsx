import { useState, useEffect } from 'react'
import PropTypes from 'prop-types'

function DocumentViewer({ documentId, filename, pdfUrl, onClose }) {
    const [loading, setLoading] = useState(true)
    const [error, setError] = useState(null)

    useEffect(() => {
        // Reset loading state when PDF URL changes
        setLoading(true)
        setError(null)
    }, [pdfUrl])

    const handleIframeLoad = () => {
        setLoading(false)
    }

    const handleIframeError = () => {
        setLoading(false)
        setError('Failed to load PDF. The file may not be available.')
    }

    const handleOverlayClick = (e) => {
        if (e.target === e.currentTarget) {
            onClose()
        }
    }

    return (
        <div className="modal-overlay" onClick={handleOverlayClick}>
            <div className="modal-content modal-pdf">
                <div className="modal-header">
                    <div>
                        <h2>{filename}</h2>
                        <p className="modal-subtitle">Document ID: {documentId}</p>
                    </div>
                    <button className="modal-close" onClick={onClose} aria-label="Close">
                        ×
                    </button>
                </div>

                <div className="modal-body">
                    {loading && (
                        <div className="pdf-loading">
                            <span className="loading"></span>
                            <p>Loading PDF...</p>
                        </div>
                    )}

                    {error && (
                        <div className="alert alert-error">
                            <strong>Error:</strong> {error}
                        </div>
                    )}

                    {pdfUrl && (
                        <iframe
                            src={pdfUrl}
                            className="pdf-viewer"
                            title={filename}
                            onLoad={handleIframeLoad}
                            onError={handleIframeError}
                            style={{ display: loading ? 'none' : 'block' }}
                        />
                    )}
                </div>

                <div className="modal-footer">
                    <a
                        href={pdfUrl}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="btn btn-primary"
                    >
                        Open in New Tab
                    </a>
                    <button className="btn btn-secondary" onClick={onClose}>
                        Close
                    </button>
                </div>
            </div>
        </div>
    )
}

DocumentViewer.propTypes = {
    documentId: PropTypes.string.isRequired,
    filename: PropTypes.string.isRequired,
    pdfUrl: PropTypes.string.isRequired,
    onClose: PropTypes.func.isRequired
}

export default DocumentViewer
