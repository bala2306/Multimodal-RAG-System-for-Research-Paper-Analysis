import axios from 'axios';

const api = axios.create({
    baseURL: '/api/v1',
    headers: {
        'Content-Type': 'application/json',
    },
});

export const uploadBasic = async (file, onProgress) => {
    const formData = new FormData();
    formData.append('file', file);

    const response = await axios.post('/api/v1/basic/upload', formData, {
        headers: {
            'Content-Type': 'multipart/form-data',
        },
        onUploadProgress: (progressEvent) => {
            if (onProgress) {
                const percentCompleted = Math.round((progressEvent.loaded * 100) / progressEvent.total);
                onProgress(percentCompleted);
            }
        },
    });

    return response.data;
};

export const uploadAdvanced = async (file, onProgress) => {
    const formData = new FormData();
    formData.append('file', file);

    const response = await axios.post('/api/v1/advanced/upload', formData, {
        headers: {
            'Content-Type': 'multipart/form-data',
        },
        onUploadProgress: (progressEvent) => {
            if (onProgress) {
                const percentCompleted = Math.round((progressEvent.loaded * 100) / progressEvent.total);
                onProgress(percentCompleted);
            }
        },
    });

    return response.data;
};

export const queryBasic = async (query, topK = 5) => {
    const response = await api.post('/basic/query', {
        query,
        top_k: topK,
    });

    return response.data;
};

export const queryAdvanced = async (query, topK = 10) => {
    try {
        const response = await api.post('/advanced/query', {
            query,
            top_k: topK,
        });

        return response.data;
    } catch (error) {
        console.error('Advanced query error:', error)
        throw error
    }
};

// Get basic documents list
export const getBasicDocuments = async () => {
    try {
        const response = await axios.get('/api/v1/basic/documents')
        return response.data
    } catch (error) {
        console.error('Get basic documents error:', error)
        throw error
    }
}

// Get advanced documents list
export const getAdvancedDocuments = async () => {
    try {
        const response = await axios.get('/api/v1/advanced/documents')
        return response.data
    } catch (error) {
        console.error('Get advanced documents error:', error)
        throw error
    }
}

// Get PDF URL for a document
export const getDocumentPdf = async (documentId, ingestionType = 'basic') => {
    try {
        const endpoint = ingestionType === 'basic' ? '/api/v1/basic' : '/api/v1/advanced'
        const response = await axios.get(`${endpoint}/documents/${documentId}/pdf`)
        return response.data
    } catch (error) {
        console.error('Get document PDF error:', error)
        throw error
    }
}

// Get images for an advanced document
export const getDocumentImages = async (documentId) => {
    try {
        const response = await axios.get(`/api/v1/advanced/documents/${documentId}/images`)
        return response.data
    } catch (error) {
        console.error('Get document images error:', error)
        throw error
    }
}

// Get tables for an advanced document
export const getDocumentTables = async (documentId) => {
    try {
        const response = await axios.get(`/api/v1/advanced/documents/${documentId}/tables`)
        return response.data
    } catch (error) {
        console.error('Get document tables error:', error)
        throw error
    }
}

// Delete a document (basic or advanced)
export const deleteDocument = async (documentId, ingestionType = 'basic') => {
    try {
        const endpoint = ingestionType === 'basic' ? '/api/v1/basic' : '/api/v1/advanced'
        const response = await axios.delete(`${endpoint}/documents/${documentId}`)
        return response.data
    } catch (error) {
        console.error('Delete document error:', error)
        throw error
    }
}

export default api;
