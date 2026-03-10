import { useState } from 'react'
import PropTypes from 'prop-types'

function ImageGallery({ documentId, filename, images, onClose, embedded = false }) {
    const [selectedImage, setSelectedImage] = useState(null)

    const handleOverlayClick = (e) => {
        if (e.target === e.currentTarget) {
            if (selectedImage) {
                setSelectedImage(null)
            } else if (!embedded) {
                onClose()
            }
        }
    }

    const handleImageClick = (image) => {
        setSelectedImage(image)
    }

    const handleNextImage = () => {
        if (!selectedImage) return
        const currentIndex = images.findIndex(img => img.id === selectedImage.id)
        const nextIndex = (currentIndex + 1) % images.length
        setSelectedImage(images[nextIndex])
    }

    const handlePrevImage = () => {
        if (!selectedImage) return
        const currentIndex = images.findIndex(img => img.id === selectedImage.id)
        const prevIndex = (currentIndex - 1 + images.length) % images.length
        setSelectedImage(images[prevIndex])
    }

    const galleryContent = (
        <>
            {images.length === 0 ? (
                <div className="empty-state">
                    <div className="empty-icon">🖼️</div>
                    <h3>No Images Found</h3>
                    <p>This document doesn't contain any extracted images.</p>
                </div>
            ) : (
                <div className="image-gallery-grid">
                    {images.map((image) => (
                        <div
                            key={image.id}
                            className="image-thumbnail"
                            onClick={() => handleImageClick(image)}
                        >
                            {image.image_url ? (
                                <img src={image.image_url} alt={image.description} />
                            ) : (
                                <div className="image-placeholder">
                                    <span>No Preview</span>
                                </div>
                            )}
                            <div className="image-info">
                                <span className="image-page">Page {image.page_number}</span>
                                <span className="image-type">{image.element_type}</span>
                            </div>
                            <p className="image-description">{image.description}</p>
                        </div>
                    ))}
                </div>
            )}
        </>
    )

    // If embedded, just return the content without modal wrapper
    if (embedded) {
        return (
            <>
                {galleryContent}
                {selectedImage && (
                    <div className="image-lightbox" onClick={handleOverlayClick}>
                        <button className="lightbox-close" onClick={() => setSelectedImage(null)}>
                            ×
                        </button>
                        <button className="lightbox-nav lightbox-prev" onClick={handlePrevImage}>
                            ‹
                        </button>
                        <button className="lightbox-nav lightbox-next" onClick={handleNextImage}>
                            ›
                        </button>
                        <div className="lightbox-content">
                            {selectedImage.image_url ? (
                                <img src={selectedImage.image_url} alt={selectedImage.description} />
                            ) : (
                                <div className="lightbox-placeholder">Image not available</div>
                            )}
                            <div className="lightbox-info">
                                <p className="lightbox-page">Page {selectedImage.page_number}</p>
                                <p className="lightbox-description">{selectedImage.description}</p>
                            </div>
                        </div>
                    </div>
                )}
            </>
        )
    }

    // Full modal version
    return (
        <div className="modal-overlay" onClick={handleOverlayClick}>
            <div className="modal-content modal-gallery">
                <div className="modal-header">
                    <div>
                        <h2>Images from {filename}</h2>
                        <p className="modal-subtitle">{images.length} image{images.length !== 1 ? 's' : ''} found</p>
                    </div>
                    <button className="modal-close" onClick={onClose} aria-label="Close">
                        ×
                    </button>
                </div>

                <div className="modal-body">
                    {images.length === 0 ? (
                        <div className="empty-state">
                            <div className="empty-icon">🖼️</div>
                            <h3>No Images Found</h3>
                            <p>This document doesn't contain any extracted images.</p>
                        </div>
                    ) : (
                        <div className="image-gallery-grid">
                            {images.map((image) => (
                                <div
                                    key={image.id}
                                    className="image-thumbnail"
                                    onClick={() => handleImageClick(image)}
                                >
                                    {image.image_url ? (
                                        <img src={image.image_url} alt={image.description} />
                                    ) : (
                                        <div className="image-placeholder">
                                            <span>No Preview</span>
                                        </div>
                                    )}
                                    <div className="image-info">
                                        <span className="image-page">Page {image.page_number}</span>
                                        <span className="image-type">{image.element_type}</span>
                                    </div>
                                    <p className="image-description">{image.description}</p>
                                </div>
                            ))}
                        </div>
                    )}
                </div>

                <div className="modal-footer">
                    <button className="btn btn-primary" onClick={onClose}>
                        Close
                    </button>
                </div>
            </div>

            {/* Lightbox for full-size image */}
            {selectedImage && (
                <div className="image-lightbox" onClick={handleOverlayClick}>
                    <button className="lightbox-close" onClick={() => setSelectedImage(null)}>
                        ×
                    </button>
                    <button className="lightbox-nav lightbox-prev" onClick={handlePrevImage}>
                        ‹
                    </button>
                    <button className="lightbox-nav lightbox-next" onClick={handleNextImage}>
                        ›
                    </button>
                    <div className="lightbox-content">
                        {selectedImage.image_url ? (
                            <img src={selectedImage.image_url} alt={selectedImage.description} />
                        ) : (
                            <div className="lightbox-placeholder">Image not available</div>
                        )}
                        <div className="lightbox-info">
                            <p className="lightbox-page">Page {selectedImage.page_number}</p>
                            <p className="lightbox-description">{selectedImage.description}</p>
                        </div>
                    </div>
                </div>
            )}
        </div>
    )
}

ImageGallery.propTypes = {
    documentId: PropTypes.string.isRequired,
    filename: PropTypes.string.isRequired,
    images: PropTypes.arrayOf(PropTypes.shape({
        id: PropTypes.string.isRequired,
        element_type: PropTypes.string.isRequired,
        page_number: PropTypes.number.isRequired,
        image_url: PropTypes.string,
        description: PropTypes.string.isRequired,
        metadata: PropTypes.object
    })).isRequired,
    onClose: PropTypes.func.isRequired,
    embedded: PropTypes.bool
}

export default ImageGallery
