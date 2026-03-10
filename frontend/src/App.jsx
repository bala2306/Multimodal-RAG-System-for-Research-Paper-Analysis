import { useState } from 'react'
import BasicIngestion from './components/BasicIngestion'
import AdvancedIngestion from './components/AdvancedIngestion'
import BasicQuery from './components/BasicQuery'
import AdvancedQuery from './components/AdvancedQuery'
import BasicDocuments from './components/BasicDocuments'
import AdvancedDocuments from './components/AdvancedDocuments'

function App() {
    const [activeView, setActiveView] = useState('basic-query')

    const menuItems = [
        {
            id: 'basic-query',
            label: 'Basic Query',
            description: 'Simple text retrieval',
            component: BasicQuery
        },
        {
            id: 'advanced-query',
            label: 'Advanced Query',
            description: 'Multimodal search',
            component: AdvancedQuery
        },
        {
            id: 'basic-documents',
            label: 'Basic Documents',
            description: 'Browse basic collection',
            component: BasicDocuments
        },
        {
            id: 'advanced-documents',
            label: 'Advanced Documents',
            description: 'Browse advanced collection',
            component: AdvancedDocuments
        },
        {
            id: 'basic-ingestion',
            label: 'Basic Ingestion',
            description: 'PDF text extraction',
            component: BasicIngestion
        },
        {
            id: 'advanced-ingestion',
            label: 'Advanced Ingestion',
            description: 'Multimodal processing',
            component: AdvancedIngestion
        },
    ]

    const ActiveComponent = menuItems.find(item => item.id === activeView)?.component
    const activeItem = menuItems.find(item => item.id === activeView)

    return (
        <div className="app-layout">
            {/* Sidebar */}
            <aside className="sidebar">
                <div className="sidebar-header">
                    <div className="logo">
                        <div className="logo-icon">RAG</div>
                        <div className="logo-text">
                            <h2>RAG Pipeline</h2>
                            <p>Document Intelligence</p>
                        </div>
                    </div>
                </div>

                <nav className="sidebar-nav">
                    <div className="nav-section">
                        <div className="nav-section-title">Query</div>
                        {menuItems.slice(0, 2).map(item => (
                            <button
                                key={item.id}
                                className={`nav-item ${activeView === item.id ? 'active' : ''}`}
                                onClick={() => setActiveView(item.id)}
                            >
                                <div className="nav-content">
                                    <span className="nav-label">{item.label}</span>
                                    <span className="nav-description">{item.description}</span>
                                </div>
                            </button>
                        ))}
                    </div>

                    <div className="nav-section">
                        <div className="nav-section-title">Documents</div>
                        {menuItems.slice(2, 4).map(item => (
                            <button
                                key={item.id}
                                className={`nav-item ${activeView === item.id ? 'active' : ''}`}
                                onClick={() => setActiveView(item.id)}
                            >
                                <div className="nav-content">
                                    <span className="nav-label">{item.label}</span>
                                    <span className="nav-description">{item.description}</span>
                                </div>
                            </button>
                        ))}
                    </div>

                    <div className="nav-section">
                        <div className="nav-section-title">Ingestion</div>
                        {menuItems.slice(4, 6).map(item => (
                            <button
                                key={item.id}
                                className={`nav-item ${activeView === item.id ? 'active' : ''}`}
                                onClick={() => setActiveView(item.id)}
                            >
                                <div className="nav-content">
                                    <span className="nav-label">{item.label}</span>
                                    <span className="nav-description">{item.description}</span>
                                </div>
                            </button>
                        ))}
                    </div>
                </nav>

                <div className="sidebar-footer">
                    <div className="status-indicator">
                        <span className="status-dot"></span>
                        <span className="status-text">Backend Connected</span>
                    </div>
                </div>
            </aside>

            {/* Main Content */}
            <main className="main-content">
                <header className="content-header">
                    <div>
                        <h1>{activeItem?.label}</h1>
                        <p>{activeItem?.description}</p>
                    </div>
                </header>

                <div className="content-body">
                    {ActiveComponent && <ActiveComponent />}
                </div>
            </main>
        </div>
    )
}

export default App
