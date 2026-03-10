# RAG System Improvements

## Overview
This document describes the improvements made to the RAG system based on evaluation results that showed the system was underperforming (9-17% accuracy). The improvements target the root causes identified in the evaluation analysis.

## Evaluation Results Summary

### Before Improvements
| Metric | No RAG | Basic RAG | Advanced RAG |
|--------|---------|-----------|--------------|
| **Accuracy** | 16.92% | 10.81% | 9.43% |
| **Keyword Overlap** | 53.16% | 41.79% | 27.01% |
| **Answer Relevancy** | 94.62% | 56.41% | 36.41% |
| **Faithfulness** | N/A | 56.15% | 58.21% |
| **Latency** | 1,881ms | 3,486ms | 3,665ms |

### Key Problems Identified
1. **RAG Degradation Paradox**: RAG systems performed worse than no RAG
2. **Faithfulness-Accuracy Disconnect**: High faithfulness to wrong/incomplete context
3. **Relevancy Collapse**: Retrieved context making answers less relevant
4. **Poor Retrieval Quality**: Top-k too low, context doesn't contain answers
5. **Query-Document Mismatch**: Dense retrieval missing exact phrase matches

---

## Improvements Implemented

### 1. Hybrid Retrieval (BM25 + Dense Vectors)
**Problem**: Dense retrieval alone missed exact phrase matches like "BERT stands for..."

**Solution**: Implemented hybrid search combining:
- **BM25 (keyword-based)**: Captures exact phrase and term matches
- **Dense vectors (semantic)**: Captures meaning and context
- **Reciprocal Rank Fusion (RRF)**: Combines results from both methods

**Files**:
- `app/services/retrieval/bm25_retriever.py` - BM25 implementation
- `app/services/retrieval/hybrid_retriever.py` - Hybrid search with RRF

**Expected Impact**: +3-5% accuracy improvement

**Configuration**:
```env
USE_HYBRID_SEARCH=true
BM25_WEIGHT=0.5  # Weight for BM25 in linear combination
RRF_K=60  # Reciprocal Rank Fusion parameter
```

---

### 2. Cross-Encoder Re-Ranking
**Problem**: Initial retrieval returned false positives; need better precision

**Solution**: Added cross-encoder re-ranking after initial retrieval
- Uses `cross-encoder/ms-marco-MiniLM-L-6-v2` model
- Re-scores query-document pairs with full attention
- Filters out low-quality matches

**Files**:
- `app/services/retrieval/reranker.py`

**Expected Impact**: +2-4% accuracy improvement

**Configuration**:
```env
USE_RERANKING=true
RERANKER_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2
RERANK_TOP_K=10  # Final number of results after re-ranking
```

---

### 3. Better Embedding Model
**Problem**: `all-MiniLM-L6-v2` (384-dim) is general-purpose, not domain-optimized

**Solution**: Upgraded to `all-mpnet-base-v2`
- **Dimensions**: 384 → 768 (2x richer representations)
- **Performance**: Better semantic understanding
- **Trade-off**: Slightly slower but more accurate

**Expected Impact**: +2-3% accuracy improvement

**Configuration**:
```env
EMBEDDING_MODEL=sentence-transformers/all-mpnet-base-v2
EMBEDDING_DIMENSION=768
```

**Migration Required**: Existing Qdrant collections must be recreated
```bash
python database/migrate_collections.py --force
# Then re-upload all documents
```

---

### 4. Increased top_k Retrieval
**Problem**: top_k=5 was too low, missing relevant chunks

**Solution**: Increased retrieval parameters
- **Basic RAG**: 5 → 15 (+200%)
- **Advanced RAG**: 10 → 20 (+100%)
- Retrieve more initially, then filter with re-ranking

**Expected Impact**: +1-2% accuracy improvement

**Configuration**:
```env
BASIC_TOP_K=15
ADVANCED_TOP_K=20
```

---

### 5. Query Expansion for Acronyms
**Problem**: Queries ask "What does BERT stand for?" but docs use full name

**Solution**: Automatic query expansion
- Detects acronyms in queries (BERT, ELMo, RAG, etc.)
- Expands with full names
- Maps 50+ common NLP/ML acronyms

**Files**:
- `app/services/retrieval/query_expander.py`

**Expected Impact**: +2-3% accuracy on acronym questions

**Example**:
```python
# Query: "What does BERT stand for?"
# Expanded: "What does BERT stand for? Bidirectional Encoder Representations from Transformers"
```

---

### 6. Relaxed LLM Prompts
**Problem**: Prompts too strict - "Only use context" → high faithfulness, low accuracy

**Solution**: New prompt strategy
- **Old**: "Answer ONLY from provided context"
- **New**: "Prioritize context, but use knowledge when context is incomplete"
- Allows LLM to combine retrieved context + general knowledge
- Clearly marks knowledge-based vs. context-based information

**Files**:
- `app/services/llm/llm_service.py` (lines 68-79, 189-192)

**Expected Impact**: +3-5% accuracy improvement

**Prompt Changes**:
```python
# System prompt now includes:
"Answer questions by PRIORITIZING the provided document context,
but you may also use your general knowledge when:
  - The context is incomplete or doesn't fully answer the question
  - The question asks for well-known facts that aren't in the context
  - Additional context from your knowledge enhances understanding"
```

---

### 7. Improved Chunking Strategy
**Problem**: Fixed 1000-char chunks split context too aggressively

**Solution**: Larger chunks with more overlap
- **Basic RAG**: 1000 → 1500 chars (+50%)
- **Overlap**: 200 → 300 chars (+50%)
- **Advanced RAG**: 1500 → 2000 max chars (+33%)
- **Min size**: 500 → 800 chars (+60%)

**Files**:
- `app/utils/chunking.py`

**Expected Impact**: +1-2% accuracy (more complete context per chunk)

**Configuration** (hardcoded in FixedSizeChunker and SemanticChunker):
```python
# Basic
FixedSizeChunker(chunk_size=1500, overlap=300)

# Advanced
SemanticChunker(min_chunk_size=800, max_chunk_size=2000, table_context_window=800)
```

---

## Cumulative Expected Impact

| Improvement | Expected Gain |
|-------------|---------------|
| Hybrid Retrieval (BM25 + Dense) | +3-5% |
| Cross-Encoder Re-Ranking | +2-4% |
| Better Embeddings (768-dim) | +2-3% |
| Increased top_k | +1-2% |
| Query Expansion | +2-3% |
| Relaxed Prompts | +3-5% |
| Larger Chunks | +1-2% |
| **Total Estimated** | **+14-24%** |

**Conservative Projection**: Basic RAG: 10.81% → **20-25%** accuracy

---

## Setup and Migration

### 1. Install New Dependencies
```bash
pip install -r requirements.txt
```

New dependencies added:
- `rank-bm25==0.2.2` (for BM25 retrieval)
- Cross-encoder model from sentence-transformers (auto-downloaded)

### 2. Update Environment Variables
Add to your `.env` file:
```env
# Embedding (IMPORTANT: Changed from 384 to 768)
EMBEDDING_MODEL=sentence-transformers/all-mpnet-base-v2
EMBEDDING_DIMENSION=768

# Retrieval
BASIC_TOP_K=15
ADVANCED_TOP_K=20

# Hybrid Retrieval
USE_HYBRID_SEARCH=true
BM25_WEIGHT=0.5
RRF_K=60

# Re-ranking
USE_RERANKING=true
RERANKER_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2
RERANK_TOP_K=10
```

### 3. Migrate Qdrant Collections
**Option A: Force Recreate (Recommended for Testing)**
```bash
python database/migrate_collections.py --force
# Type "DELETE ALL DATA" to confirm
```

**Option B: Manual Review**
```bash
python database/migrate_collections.py
# Review migration plan without making changes
```

After migration, **re-upload all documents** to generate new 768-dim embeddings.

### 4. Test the Improvements
```bash
# Start the server
cd RAG-Pipeline
uvicorn app.main:app --reload

# Upload a test document
curl -X POST "http://localhost:8000/api/v1/basic/upload" \
  -F "file=@path/to/test.pdf"

# Query with improvements
curl -X POST "http://localhost:8000/api/v1/basic/query" \
  -H "Content-Type: application/json" \
  -d '{"query": "What does BERT stand for?", "top_k": 15}'
```

---

## Feature Flags

All improvements can be toggled via environment variables:

### Disable Hybrid Search
```env
USE_HYBRID_SEARCH=false
```
Falls back to dense vector search only

### Disable Re-Ranking
```env
USE_RERANKING=false
```
Skips cross-encoder re-ranking step

### Use Old Embedding Model
```env
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
EMBEDDING_DIMENSION=384
```
Reverts to original smaller model (not recommended)

---

## Performance Considerations

### Latency Impact
- **Hybrid Search**: +100-200ms (BM25 indexing + fusion)
- **Re-Ranking**: +300-500ms (cross-encoder inference)
- **Larger Embeddings**: +50-100ms (768-dim vs 384-dim)
- **Total Added**: ~450-800ms per query

**Trade-off**: Accuracy vs. Speed
- With all improvements: ~2.3-2.8s per query (vs. 1.9s baseline)
- **Recommendation**: Enable all for accuracy-critical applications

### Memory Impact
- **768-dim embeddings**: 2x memory per vector
- **Cross-encoder model**: ~100MB model download
- **BM25 index**: Minimal (~5-10MB for 10k chunks)

### Optimization Tips
1. **For Speed-Critical Apps**: Disable re-ranking, keep hybrid search
2. **For Accuracy-Critical Apps**: Enable all improvements
3. **For Resource-Constrained**: Use 384-dim embeddings, enable hybrid + query expansion only

---

## Architecture Changes

### New Service Layer
```
app/services/
├── retrieval/           # NEW: Retrieval components
│   ├── bm25_retriever.py
│   ├── hybrid_retriever.py
│   ├── reranker.py
│   └── query_expander.py
├── rag/                 # NEW: RAG orchestration
│   └── improved_basic_rag.py
├── embeddings/
├── llm/
├── pdf/
└── vector_store/
```

### Modified Files
- `app/core/config.py` - New config parameters
- `app/utils/chunking.py` - Larger chunk sizes
- `app/services/llm/llm_service.py` - Relaxed prompts
- `database/init_qdrant.py` - New embedding dimension
- `requirements.txt` - New dependencies

### New Files
- `app/services/retrieval/*` - All retrieval components
- `app/services/rag/improved_basic_rag.py` - Improved RAG orchestration
- `database/migrate_collections.py` - Migration script
- `IMPROVEMENTS.md` - This file

---

## Testing and Validation

### Unit Tests (TODO)
```bash
pytest tests/test_retrieval.py  # Test hybrid search
pytest tests/test_reranking.py  # Test cross-encoder
pytest tests/test_query_expansion.py  # Test acronym expansion
```

### Integration Tests
Re-run evaluation framework with same 39 questions:
```bash
cd evaluation
python run_evaluation.py --methods improved_basic_rag
```

Expected results:
- **Accuracy**: 20-25% (vs. 10.81%)
- **Keyword Overlap**: 50-55% (vs. 41.79%)
- **Answer Relevancy**: 70-80% (vs. 56.41%)

### A/B Testing
1. Keep old system running as baseline
2. Deploy improvements in parallel
3. Route 50% traffic to improved system
4. Compare metrics over 1 week

---

## Troubleshooting

### Issue: "Collection dimension mismatch"
**Cause**: Trying to insert 768-dim vectors into 384-dim collection
**Solution**: Run migration script to recreate collections

### Issue: "ModuleNotFoundError: No module named 'rank_bm25'"
**Cause**: Missing BM25 dependency
**Solution**: `pip install rank-bm25`

### Issue: Cross-encoder slow or OOM
**Cause**: Large batch re-ranking
**Solution**: Reduce `RERANK_TOP_K` or disable re-ranking

### Issue: Query expansion not working
**Cause**: Acronym not in dictionary
**Solution**: Add custom acronym:
```python
from app.services.retrieval.query_expander import QueryExpander
expander = QueryExpander()
expander.add_custom_acronym("GPT", "Generative Pre-trained Transformer")
```

---

## Future Improvements (Not Implemented)

### High Priority
1. **Context Window Optimization**: Adaptive context based on query type
2. **Query Classification**: Route factual vs. analytical queries differently
3. **Domain-Specific Embeddings**: Fine-tune embeddings on NLP/ML corpus

### Medium Priority
4. **Metadata Filtering**: Filter by document section/type
5. **Multi-Query**: Generate query variations and combine results
6. **Answer Synthesis**: Combine multiple chunks before generation

### Low Priority
7. **Caching**: Cache frequent query embeddings
8. **Async Processing**: Parallel retrieval from multiple collections
9. **Confidence Thresholding**: Return "no answer" when confidence low

---

## References

- **Evaluation Report**: `evaluation/results/evaluation_20251215_160217.json`
- **Original Accuracy**: Basic RAG: 10.81%, Advanced RAG: 9.43%
- **Target Accuracy**: 20-25% (conservative), 25-30% (optimistic)

---

## Changelog

### 2025-12-16: Major Improvements Release
- ✅ Hybrid retrieval (BM25 + Dense)
- ✅ Cross-encoder re-ranking
- ✅ Better embedding model (768-dim)
- ✅ Increased top_k retrieval
- ✅ Query expansion for acronyms
- ✅ Relaxed LLM prompts
- ✅ Larger chunking strategy
- ✅ Migration tooling
- ✅ Comprehensive documentation

---

**Status**: All improvements implemented and documented. Ready for testing and evaluation.

**Next Steps**:
1. Migrate Qdrant collections (`python database/migrate_collections.py --force`)
2. Re-upload all documents with new embeddings
3. Run evaluation framework to measure improvements
4. Compare results with baseline (target: >20% accuracy)
