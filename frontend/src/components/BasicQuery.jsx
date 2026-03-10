import { useState } from 'react'
import { queryBasic } from '../services/api'

function BasicQuery() {
    const [query, setQuery] = useState('')
    const [topK, setTopK] = useState(5)
    const [loading, setLoading] = useState(false)
    const [result, setResult] = useState(null)
    const [error, setError] = useState(null)

    const handleSubmit = async (e) => {
        e.preventDefault()

        if (!query.trim()) {
            setError('Please enter a query')
            return
        }

        setLoading(true)
        setError(null)
        setResult(null)

        try {
            const data = await queryBasic(query, topK)
            setResult(data)
        } catch (err) {
            setError(err.response?.data?.detail || err.message || 'Failed to process query')
        } finally {
            setLoading(false)
        }
    }

    return (
        <div className="card">
            <h2>🔍 Basic Query</h2>
            <p style={{ color: 'var(--text-secondary)', marginBottom: '2rem' }}>
                Simple text-based retrieval from your documents
            </p>

            <form onSubmit={handleSubmit}>
                <div className="input-group">
                    <label htmlFor="query">Your Question</label>
                    <textarea
                        id="query"
                        className="input textarea"
                        value={query}
                        onChange={(e) => setQuery(e.target.value)}
                        placeholder="What would you like to know about your documents?"
                        rows={4}
                    />
                </div>

                <div className="input-group">
                    <label htmlFor="topK">Number of Results (Top K)</label>
                    <input
                        id="topK"
                        type="number"
                        className="input"
                        value={topK}
                        onChange={(e) => setTopK(parseInt(e.target.value))}
                        min={1}
                        max={20}
                    />
                </div>

                <button type="submit" className="btn btn-primary" disabled={loading}>
                    {loading ? (
                        <>
                            <span className="loading"></span>
                            <span style={{ marginLeft: '0.5rem' }}>Processing...</span>
                        </>
                    ) : (
                        'Search'
                    )}
                </button>
            </form>

            {error && (
                <div className="alert alert-error mt-3">
                    <strong>Error:</strong> {error}
                </div>
            )}

            {result && (
                <div className="result">
                    <div className="result-header">
                        <h3>Answer</h3>
                        <span style={{ color: 'var(--text-muted)', fontSize: '0.9rem' }}>
                            {result.query_time_ms}ms
                        </span>
                    </div>

                    <div className="answer-box">
                        {result.answer}
                    </div>

                    {result.sources && result.sources.length > 0 && (
                        <div className="sources-section">
                            <h3>Sources ({result.sources.length})</h3>
                            {result.sources.map((source, index) => (
                                <div key={index} className="source-item">
                                    <div className="source-header">
                                        <div className="source-meta">
                                            <strong>{source.document_name}</strong>
                                            <span className="badge">Page {source.page_number}</span>
                                            <span className="badge score-badge">
                                                {(source.relevance_score * 100).toFixed(1)}%
                                            </span>
                                        </div>
                                    </div>
                                    <div className="source-snippet">
                                        {source.text_snippet}
                                    </div>
                                </div>
                            ))}
                        </div>
                    )}
                </div>
            )}
        </div>
    )
}

export default BasicQuery
