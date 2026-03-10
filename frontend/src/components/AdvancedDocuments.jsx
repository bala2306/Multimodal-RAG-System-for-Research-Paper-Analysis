import { useState, useEffect } from 'react'
import { getAdvancedDocuments, getDocumentPdf, getDocumentImages, getDocumentTables, deleteDocument } from '../services/api'
import CombinedDocumentViewer from './CombinedDocumentViewer'

function AdvancedDocuments() {
    const [documents, setDocuments] = useState([])
    const [insights, setInsights] = useState(null)
    const [loading, setLoading] = useState(true)
    const [error, setError] = useState(null)
    const [selectedDocument, setSelectedDocument] = useState(null)
    const [pdfUrl, setPdfUrl] = useState(null)
    const [images, setImages] = useState([])
    const [tables, setTables] = useState([])
    const [loadingContent, setLoadingContent] = useState(false)

    useEffect(() => {
        fetchDocuments()
    }, [])

    const fetchDocuments = async () => {
        try {
            setLoading(true)
            setError(null)
            const data = await getAdvancedDocuments()
            setDocuments(data.documents || [])
            setInsights(data.insights || null)
        } catch (err) {
            setError(err.message || 'Failed to fetch documents')
        } finally {
            setLoading(false)
        }
    }

    const handleDocumentClick = async (doc) => {
        try {
            setLoadingContent(true)
            setSelectedDocument(doc)

            // Fetch PDF, images, and tables in parallel
            const [pdfData, imagesData, tablesData] = await Promise.all([
                getDocumentPdf(doc.id, 'advanced'),
                getDocumentImages(doc.id),
                getDocumentTables(doc.id)
            ])

            setPdfUrl(pdfData.pdf_url)
            setImages(imagesData.images || [])
            setTables(tablesData.tables || [])
        } catch (err) {
            console.error('Failed to load document content:', err)
            alert('Failed to load document. Please try again.')
            setSelectedDocument(null)
        } finally {
            setLoadingContent(false)
        }
    }

    const handleCloseViewer = () => {
        setSelectedDocument(null)
        setPdfUrl(null)
        setImages([])
        setTables([])
    }

    const handleDeleteDocument = async (e, doc) => {
        // Stop event propagation to prevent opening the document viewer
        e.stopPropagation()

        // Confirm deletion
        const confirmDelete = window.confirm(
            `Are you sure you want to delete "${doc.filename}"?\n\nThis will permanently remove:\n- The document and all its data\n- All text chunks and embeddings\n- All visual elements (${doc.visual_elements_count} items)\n- All vectors from the database\n- All stored files\n\nThis action cannot be undone.`
        )

        if (!confirmDelete) {
            return
        }

        try {
            // Call delete API
            await deleteDocument(doc.id, 'advanced')

            // Refresh the documents list
            await fetchDocuments()

            // Show success message
            alert(`Document "${doc.filename}" deleted successfully`)
        } catch (err) {
            console.error('Failed to delete document:', err)
            alert(`Failed to delete document: ${err.message || 'Unknown error'}`)
        }
    }

    const formatDate = (dateString) => {
        const date = new Date(dateString)
        return date.toLocaleDateString('en-US', {
            year: 'numeric',
            month: 'short',
            day: 'numeric'
        })
    }

    const getStatusBadge = (status) => {
        const statusMap = {
            completed: { label: 'Completed', class: 'status-completed' },
            processing: { label: 'Processing', class: 'status-processing' },
            failed: { label: 'Failed', class: 'status-failed' }
        }
        const statusInfo = statusMap[status] || { label: status, class: '' }
        return <span className={`status-badge ${statusInfo.class}`}>{statusInfo.label}</span>
    }

    if (loading) {
        return (
            <div className="card">
                <div className="loading-container">
                    <span className="loading"></span>
                    <p>Loading documents...</p>
                </div>
            </div>
        )
    }

    if (error) {
        return (
            <div className="card">
                <div className="alert alert-error">
                    <strong>Error:</strong> {error}
                </div>
            </div>
        )
    }

    if (documents.length === 0) {
        return (
            <div className="card">
                <div className="empty-state">
                    <div className="empty-icon">📊</div>
                    <h3>No Documents Yet</h3>
                    <p>Upload your first PDF using the Advanced Ingestion feature to get started.</p>
                </div>
            </div>
        )
    }

    return (
        <>
            <div className="card">
                {insights && (
                    <div className="insights-panel">
                        <h3>Collection Insights</h3>
                        <div className="insights-grid insights-grid-advanced">
                            <div className="insight-item">
                                <div className="insight-value">{insights.total_documents}</div>
                                <div className="insight-label">Total Documents</div>
                            </div>
                            <div className="insight-item">
                                <div className="insight-value">{insights.total_pages}</div>
                                <div className="insight-label">Total Pages</div>
                            </div>
                            <div className="insight-item">
                                <div className="insight-value">{insights.total_chunks}</div>
                                <div className="insight-label">Text Chunks</div>
                            </div>
                            <div className="insight-item">
                                <div className="insight-value">{insights.total_visual_elements}</div>
                                <div className="insight-label">Visual Elements</div>
                            </div>
                            <div className="insight-item">
                                <div className="insight-value">{insights.total_tables}</div>
                                <div className="insight-label">Tables</div>
                            </div>
                            <div className="insight-item">
                                <div className="insight-value">{insights.total_images}</div>
                                <div className="insight-label">Images</div>
                            </div>
                        </div>
                    </div>
                )}

                <div className="document-grid">
                    {documents.map((doc) => (
                        <div
                            key={doc.id}
                            className="document-card document-card-advanced"
                            onClick={() => handleDocumentClick(doc)}
                            style={{ cursor: 'pointer', position: 'relative' }}
                        >
                            <button
                                className="delete-button"
                                onClick={(e) => handleDeleteDocument(e, doc)}
                                title="Delete document"
                                aria-label="Delete document"
                            >
                                ✕
                            </button>

                            <div className="document-header">
                                <div className="document-icon">PDF</div>
                                <div className="document-title">
                                    <h4>{doc.filename}</h4>
                                    <span className="document-date">{formatDate(doc.upload_date)}</span>
                                </div>
                            </div>

                            <div className="document-stats">
                                <div className="stat-item">
                                    <span className="stat-label">Pages</span>
                                    <span className="stat-value">{doc.total_pages}</span>
                                </div>
                                <div className="stat-item">
                                    <span className="stat-label">Chunks</span>
                                    <span className="stat-value">{doc.chunks_count}</span>
                                </div>
                                <div className="stat-item">
                                    <span className="stat-label">Tables</span>
                                    <span className="stat-value">{doc.tables_extracted || 0}</span>
                                </div>
                                <div className="stat-item">
                                    <span className="stat-label">Images</span>
                                    <span className="stat-value">{doc.images_extracted || 0}</span>
                                </div>
                                <div className="stat-item">
                                    <span className="stat-label">Visual</span>
                                    <span className="stat-value">{doc.visual_elements_count || 0}</span>
                                </div>
                                <div className="stat-item">
                                    <span className="stat-label">Status</span>
                                    {getStatusBadge(doc.processing_status)}
                                </div>
                            </div>
                        </div>
                    ))}
                </div>
            </div>

            {selectedDocument && pdfUrl && !loadingContent && (
                <CombinedDocumentViewer
                    documentId={selectedDocument.id}
                    filename={selectedDocument.filename}
                    pdfUrl={pdfUrl}
                    images={images}
                    tables={tables}
                    onClose={handleCloseViewer}
                />
            )}

            {loadingContent && (
                <div className="modal-overlay">
                    <div className="loading-container">
                        <span className="loading"></span>
                        <p>Loading document...</p>
                    </div>
                </div>
            )}
        </>
    )
}

export default AdvancedDocuments
