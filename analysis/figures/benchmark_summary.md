# Embedding Overhead Benchmark Results

Generated: 2025-12-21T17:55:01.989629

Number of documents: 200

Device: cpu


## Embedding Models (Used in Baseline Topic Models)

| Model | Size (MB) | Docs/sec | ms/doc | Overhead |
|-------|-----------|----------|--------|----------|
| Alibaba-NLP/gte-large-en-v1.5 | 1664.2 | 9.8 | 102.50 | 1.00x |

## LLM Models (Our Approach)

| Model | Size (MB) | Docs/sec | ms/doc | Overhead |
|-------|-----------|----------|--------|----------|
| baidu/ERNIE-4.5-0.3B-PT | 1376.1 | 9.3 | 107.65 | 1.05x |

## Key Findings

- **Reference baseline**: Alibaba-NLP/gte-large-en-v1.5 (used in ZeroShotTM/CombinedTM)
- **Our smallest LLM**: baidu/ERNIE-4.5-0.3B-PT
- **Overhead ratio**: 1.05x (LLM vs baseline embedding)

### Interpretation
- The small ERNIE-0.3B model adds only 1.05x overhead compared to Sentence-BERT embeddings
- This is a modest increase considering the LLM provides richer semantic signals for topic modeling
- For a dataset of 10,000 documents, processing would take approximately:
  - Baseline (Sentence-BERT): 1025.0 seconds
  - Our approach (ERNIE-0.3B): 1076.5 seconds