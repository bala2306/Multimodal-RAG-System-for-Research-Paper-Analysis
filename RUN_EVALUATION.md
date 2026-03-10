# How to Run Full Evaluation

## Overview
The evaluation framework compares three methods:
1. **No RAG** - Direct LLM answers (no retrieval)
2. **Basic RAG** - Your basic RAG pipeline
3. **Advanced RAG** - Your advanced RAG pipeline with multimodal support

## Prerequisites

### 1. Migrate to New Embeddings (CRITICAL)
Before running evaluation, you must migrate to the new 768-dim embeddings:

```bash
# Recreate Qdrant collections
cd /Users/bala/NLP_Project_RAG/RAG-Pipeline
python database/migrate_collections.py --force
# Type "DELETE ALL DATA" to confirm
```

### 2. Upload Test Documents
You need to upload the documents that contain answers to the test questions:

```bash
# Start the API server
uvicorn app.main:app --reload

# In another terminal, upload your test documents
# (The documents used for the original evaluation)
curl -X POST "http://localhost:8000/api/v1/basic/upload" \
  -F "file=@path/to/test_document.pdf"

curl -X POST "http://localhost:8000/api/v1/advanced/upload" \
  -F "file=@path/to/test_document.pdf"
```

**Note**: The evaluation queries documents via the API, so they must be uploaded first.

### 3. Keep API Server Running
The evaluation script calls `http://localhost:8000` for RAG methods.

## Running the Evaluation

### Full Evaluation (All 39 Questions)
```bash
cd /Users/bala/NLP_Project_RAG/RAG-Pipeline

# Run all three methods with improved settings
python evaluation/run_evaluation.py \
  --top-k 15 \
  --methods no_rag basic_rag advanced_rag
```

**What happens**:
- Evaluates all 39 questions from `evaluation/datasets/test_queries.json`
- Tests all three methods (No RAG, Basic RAG, Advanced RAG)
- Uses top-k=15 (your new improved setting, vs. old 5)
- Saves results to `evaluation/results/evaluation_TIMESTAMP.json`
- Prints summary at the end

**Expected time**: ~20-30 minutes (39 questions × 3 methods = 117 queries)

---

### Quick Test (3 Questions Only)
To verify everything works before running the full evaluation:

```bash
python evaluation/run_evaluation.py --dry-run
```

This runs only 3 questions for all methods (~2 minutes).

---

### Test Only Improved Basic RAG
To compare only your improved Basic RAG vs. the baseline:

```bash
python evaluation/run_evaluation.py \
  --top-k 15 \
  --methods no_rag basic_rag
```

---

### Test Specific Number of Questions
```bash
# Test first 10 questions
python evaluation/run_evaluation.py --limit 10 --top-k 15
```

---

## Command-Line Options

```bash
python evaluation/run_evaluation.py [OPTIONS]

Options:
  --dataset, -d PATH         Path to test dataset JSON
                             (default: evaluation/datasets/test_queries.json)

  --output, -o DIR           Output directory for results
                             (default: evaluation/results)

  --top-k, -k NUM            Top-K for retrieval (default: 5)
                             RECOMMENDED: Use 15 for improved system

  --methods, -m METHOD...    Methods to evaluate
                             Options: no_rag basic_rag advanced_rag
                             (default: all three)

  --limit, -l NUM            Limit number of questions
                             Example: --limit 10 (test first 10)

  --dry-run                  Run on first 3 questions only
                             (quick test)

  -h, --help                 Show help message
```

---

## Understanding the Results

### Output Files
Results are saved in `evaluation/results/`:
- `evaluation_TIMESTAMP.json` - Full results with all questions
- `summary_TIMESTAMP.json` - Just the aggregated metrics

### Metrics Explained

**Accuracy** (0-1):
- Semantic similarity between generated and expected answer
- Uses sentence-transformers cosine similarity
- Higher = better

**Keyword Overlap** (0-1):
- Fraction of expected keywords found in answer
- Simple word matching
- Higher = better

**Faithfulness** (0-1, RAG only):
- Whether answer is grounded in retrieved context
- Checks if answer content is supported by sources
- Higher = better (but can be high even if answer is wrong!)

**Answer Relevancy** (0-1):
- Whether answer addresses the question
- No RAG should score ~95%
- RAG systems often score lower if context is irrelevant
- Higher = better

**Latency** (milliseconds):
- Time to generate answer
- Lower = faster

**Calibration Error** (0-1, No RAG only):
- How well confidence scores match actual accuracy
- Lower = better (0 = perfect calibration)

---

## Expected Results

### Before Improvements (Your Last Run)
```
Method         Accuracy  Keyword   Latency   Relevancy
-------------------------------------------------------
No RAG         16.92%    53.16%    1,881ms   94.62%
Basic RAG      10.81%    41.79%    3,486ms   56.41%
Advanced RAG    9.43%    27.01%    3,665ms   36.41%
```

### After Improvements (Expected)
```
Method         Accuracy  Keyword   Latency   Relevancy  Change
--------------------------------------------------------------------
No RAG         16.92%    53.16%    1,881ms   94.62%     (baseline)
Basic RAG      20-25%    50-55%    3,800ms   70-80%     +10-14% ✓
Advanced RAG   20-25%    50-55%    4,000ms   70-80%     +11-16% ✓
```

**Key Improvements to Look For**:
1. ✅ **Accuracy**: Should increase by 10-14% (absolute)
2. ✅ **Keyword Overlap**: Should improve to 50-55%
3. ✅ **Answer Relevancy**: Should increase significantly (70-80%)
4. ⚠️ **Latency**: Will increase slightly (+300-400ms) due to hybrid search + re-ranking

---

## Troubleshooting

### Error: "API server not running"
**Solution**: Start the API server in another terminal:
```bash
cd /Users/bala/NLP_Project_RAG/RAG-Pipeline
uvicorn app.main:app --reload
```

### Error: "No results from dense search"
**Cause**: Documents not uploaded or wrong embeddings
**Solution**:
1. Upload test documents via API
2. Verify Qdrant collections have vectors:
   ```bash
   python database/init_qdrant.py
   ```

### Low Accuracy (< 15%)
**Possible causes**:
1. Documents uploaded with old 384-dim embeddings
2. BM25 index not built (hybrid search disabled)
3. Query expansion not working

**Solution**:
```bash
# Delete old collections and re-upload
python database/migrate_collections.py --force
# Re-upload all documents with new embeddings
```

### Evaluation Too Slow
**Solution**: Run dry-run first to verify, then run full:
```bash
# Quick test (3 questions)
python evaluation/run_evaluation.py --dry-run

# If that works, run full evaluation
python evaluation/run_evaluation.py --top-k 15
```

---

## Comparing Results

### Compare with Previous Run
Your previous results are in:
```
evaluation/results/evaluation_20251215_160217.json
```

To compare:
```bash
# Run new evaluation
python evaluation/run_evaluation.py --top-k 15

# Compare files manually or use this script:
python -c "
import json
old = json.load(open('evaluation/results/evaluation_20251215_160217.json'))
new = json.load(open('evaluation/results/evaluation_XXXXXX_XXXXXX.json'))  # Replace with new timestamp

print('ACCURACY COMPARISON:')
for method in ['no_rag', 'basic_rag', 'advanced_rag']:
    old_acc = old['summary'][method]['mean_accuracy']
    new_acc = new['summary'][method]['mean_accuracy']
    delta = (new_acc - old_acc) * 100
    print(f'{method}: {old_acc*100:.1f}% → {new_acc*100:.1f}% ({delta:+.1f}%)')
"
```

---

## Re-running After Changes

If you make more improvements:

1. **Re-migrate collections** (if embedding model changed):
   ```bash
   python database/migrate_collections.py --force
   ```

2. **Re-upload documents**:
   ```bash
   # Upload via API (server must be running)
   curl -X POST "http://localhost:8000/api/v1/basic/upload" -F "file=@doc.pdf"
   ```

3. **Run evaluation**:
   ```bash
   python evaluation/run_evaluation.py --top-k 15
   ```

4. **Compare with baseline**:
   Check `evaluation/results/evaluation_TIMESTAMP.json`

---

## Quick Checklist

Before running full evaluation, verify:

- [ ] API server running (`uvicorn app.main:app --reload`)
- [ ] Qdrant collections migrated to 768-dim
- [ ] Test documents uploaded (via `/api/v1/basic/upload` and `/api/v1/advanced/upload`)
- [ ] `.env` file has new settings:
  ```env
  EMBEDDING_MODEL=sentence-transformers/all-mpnet-base-v2
  EMBEDDING_DIMENSION=768
  BASIC_TOP_K=15
  ADVANCED_TOP_K=20
  USE_HYBRID_SEARCH=true
  USE_RERANKING=true
  ```
- [ ] Dependencies installed (`pip install -r requirements.txt`)

Once all checked, run:
```bash
python evaluation/run_evaluation.py --dry-run  # Quick test
python evaluation/run_evaluation.py --top-k 15  # Full evaluation
```

---

## Example Session

```bash
# Terminal 1: Start API server
cd /Users/bala/NLP_Project_RAG/RAG-Pipeline
uvicorn app.main:app --reload

# Terminal 2: Run evaluation
cd /Users/bala/NLP_Project_RAG/RAG-Pipeline

# Quick test first
python evaluation/run_evaluation.py --dry-run

# If successful, run full evaluation
python evaluation/run_evaluation.py --top-k 15

# Results will be in:
# evaluation/results/evaluation_TIMESTAMP.json
```

---

**Ready to run?** Start with `--dry-run` to verify everything works!
