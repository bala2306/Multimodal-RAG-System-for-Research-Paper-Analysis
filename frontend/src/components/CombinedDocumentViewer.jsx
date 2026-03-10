import { useState } from 'react'
import PropTypes from 'prop-types'
import ImageGallery from './ImageGallery'

function CombinedDocumentViewer({ documentId, filename, pdfUrl, images, tables, onClose }) {
    const [activeTab, setActiveTab] = useState('pdf')

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

                {/* Tab Navigation */}
                <div className="modal-tabs">
                    <button
                        className={`modal-tab ${activeTab === 'pdf' ? 'active' : ''}`}
                        onClick={() => setActiveTab('pdf')}
                    >
                        PDF Document
                    </button>
                    <button
                        className={`modal-tab ${activeTab === 'images' ? 'active' : ''}`}
                        onClick={() => setActiveTab('images')}
                    >
                        Images ({images.length})
                    </button>
                    <button
                        className={`modal-tab ${activeTab === 'tables' ? 'active' : ''}`}
                        onClick={() => setActiveTab('tables')}
                    >
                        Tables ({tables.length})
                    </button>
                </div>

                <div className="modal-body">
                    {activeTab === 'pdf' && (
                        <>
                            {pdfUrl && (
                                <iframe
                                    src={pdfUrl}
                                    className="pdf-viewer"
                                    title={filename}
                                />
                            )}
                        </>
                    )}

                    {activeTab === 'images' && (
                        <div className="modal-gallery-content">
                            {images.length === 0 ? (
                                <div className="empty-state">
                                    <div className="empty-icon">🖼️</div>
                                    <h3>No Images Found</h3>
                                    <p>This document doesn't contain any extracted images.</p>
                                </div>
                            ) : (
                                <ImageGallery
                                    documentId={documentId}
                                    filename={filename}
                                    images={images}
                                    onClose={onClose}
                                    embedded={true}
                                />
                            )}
                        </div>
                    )}

                    {activeTab === 'tables' && (
                        <div className="modal-gallery-content">
                            {tables.length === 0 ? (
                                <div className="empty-state">
                                    <div className="empty-icon">📊</div>
                                    <h3>No Tables Found</h3>
                                    <p>This document doesn't contain any extracted tables.</p>
                                </div>
                            ) : (
                                <div className="tables-list">
                                    {tables.map((table, index) => (
                                        <div key={table.id} className="table-item">
                                            <div className="table-header">
                                                <h4>Table {index + 1} - Page {table.page_number}</h4>
                                                <p className="table-description">{table.description}</p>
                                            </div>
                                            <div className="table-image-container">
                                                {table.image_url ? (
                                                    <img
                                                        src={table.image_url}
                                                        alt={`Table ${index + 1}`}
                                                        className="table-image"
                                                    />
                                                ) : (
                                                    <div className="table-placeholder">
                                                        <p>No table image available</p>
                                                    </div>
                                                )}
                                            </div>
                                        </div>
                                    ))}
                                </div>
                            )}
                        </div>
                    )}
                </div>

                <div className="modal-footer">
                    {activeTab === 'pdf' && (
                        <a
                            href={pdfUrl}
                            target="_blank"
                            rel="noopener noreferrer"
                            className="btn btn-primary"
                        >
                            Open PDF in New Tab
                        </a>
                    )}
                    <button className="btn btn-secondary" onClick={onClose}>
                        Close
                    </button>
                </div>
            </div>
        </div>
    )
}

CombinedDocumentViewer.propTypes = {
    documentId: PropTypes.string.isRequired,
    filename: PropTypes.string.isRequired,
    pdfUrl: PropTypes.string.isRequired,
    images: PropTypes.array.isRequired,
    tables: PropTypes.array.isRequired,
    onClose: PropTypes.func.isRequired
}

export default CombinedDocumentViewer
