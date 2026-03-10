-- Enable UUID extension
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Documents Table
CREATE TABLE IF NOT EXISTS documents (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    filename TEXT NOT NULL,
    file_path TEXT,
    upload_date TIMESTAMP DEFAULT NOW(),
    total_pages INTEGER,
    processing_status TEXT CHECK (processing_status IN ('pending', 'processing', 'completed', 'failed')),
    ingestion_type TEXT NOT NULL CHECK (ingestion_type IN ('basic', 'advanced')),
    metadata JSONB,
    error_message TEXT,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_documents_ingestion_type ON documents(ingestion_type);
CREATE INDEX IF NOT EXISTS idx_documents_status ON documents(processing_status);
CREATE INDEX IF NOT EXISTS idx_documents_upload_date ON documents(upload_date DESC);

-- Chunks Table
CREATE TABLE IF NOT EXISTS chunks (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    document_id UUID REFERENCES documents(id) ON DELETE CASCADE,
    chunk_text TEXT NOT NULL,
    chunk_index INTEGER NOT NULL,
    page_number INTEGER,
    parent_chunk_id UUID REFERENCES chunks(id) ON DELETE SET NULL,
    chunk_type TEXT CHECK (chunk_type IN ('text', 'table_context', 'image_context')),
    ingestion_type TEXT NOT NULL CHECK (ingestion_type IN ('basic', 'advanced')),
    visual_element_ids UUID[],
    metadata JSONB,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_chunks_document_id ON chunks(document_id);
CREATE INDEX IF NOT EXISTS idx_chunks_ingestion_type ON chunks(ingestion_type);
CREATE INDEX IF NOT EXISTS idx_chunks_parent_chunk_id ON chunks(parent_chunk_id);

-- Visual Elements Table (Advanced RAG Only)
CREATE TABLE IF NOT EXISTS visual_elements (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    document_id UUID REFERENCES documents(id) ON DELETE CASCADE,
    element_type TEXT NOT NULL CHECK (element_type IN ('table', 'image', 'chart', 'figure')),
    page_number INTEGER,
    bounding_box JSONB,
    file_path TEXT,
    table_data JSONB,
    table_markdown TEXT,
    text_annotation TEXT NOT NULL,
    related_chunk_ids UUID[],
    ingestion_type TEXT DEFAULT 'advanced',
    metadata JSONB,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_visual_elements_document_id ON visual_elements(document_id);
CREATE INDEX IF NOT EXISTS idx_visual_elements_element_type ON visual_elements(element_type);
CREATE INDEX IF NOT EXISTS idx_visual_elements_page_number ON visual_elements(page_number);

-- Embeddings Table
CREATE TABLE IF NOT EXISTS embeddings (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    chunk_id UUID REFERENCES chunks(id) ON DELETE CASCADE,
    visual_element_id UUID REFERENCES visual_elements(id) ON DELETE CASCADE,
    collection_name TEXT NOT NULL,
    vector_id TEXT NOT NULL,
    embedding_model TEXT NOT NULL,
    ingestion_type TEXT NOT NULL CHECK (ingestion_type IN ('basic', 'advanced')),
    created_at TIMESTAMP DEFAULT NOW(),
    CONSTRAINT chunk_or_visual CHECK (
        (chunk_id IS NOT NULL AND visual_element_id IS NULL) OR
        (chunk_id IS NULL AND visual_element_id IS NOT NULL)
    )
);

CREATE INDEX IF NOT EXISTS idx_embeddings_chunk_id ON embeddings(chunk_id);
CREATE INDEX IF NOT EXISTS idx_embeddings_visual_element_id ON embeddings(visual_element_id);
CREATE INDEX IF NOT EXISTS idx_embeddings_collection_name ON embeddings(collection_name);
CREATE INDEX IF NOT EXISTS idx_embeddings_ingestion_type ON embeddings(ingestion_type);
CREATE INDEX IF NOT EXISTS idx_embeddings_vector_id ON embeddings(vector_id);

-- Query Logs Table (Optional - for analytics)
CREATE TABLE IF NOT EXISTS query_logs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    query_text TEXT NOT NULL,
    ingestion_type TEXT NOT NULL CHECK (ingestion_type IN ('basic', 'advanced')),
    collection_searched TEXT[],
    retrieved_chunk_ids UUID[],
    response_time_ms INTEGER,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_query_logs_ingestion_type ON query_logs(ingestion_type);
CREATE INDEX IF NOT EXISTS idx_query_logs_created_at ON query_logs(created_at DESC);

-- Update timestamp trigger function
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Apply trigger to documents table
CREATE TRIGGER update_documents_updated_at BEFORE UPDATE ON documents
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
