# Performance Experiments

## Project

**Bhagavatham AI Agent**

This document records all performance experiments performed on the RAG pipeline. Every optimization is benchmarked before being accepted. Changes are retained only if they produce measurable improvements without affecting correctness.

---

# Test Environment

| Item               | Value                               |
| ------------------ | ----------------------------------- |
| Operating System   | Windows 11 (64-bit)                 |
| Processor          | Intel Core i7-10510U CPU @ 1.80 GHz |
| Physical Cores     | 4                                   |
| Logical Threads    | 8                                   |
| RAM                | 16 GB                               |
| PyTorch Threads    | 4                                   |
| Embedding Model    | all-MiniLM-L6-v2                    |
| Embedding Provider | Sentence Transformers               |
| Vector Database    | ChromaDB                            |

---

# Dataset

| Metric              |     Value |
| ------------------- | --------: |
| Documents           |         2 |
| Chunks              |     6,889 |
| Characters          | 4,477,455 |
| Embedding Dimension |       384 |

---

# Baseline Performance

| Stage     |  Time (sec) |
| --------- | ----------: |
| Corpus    |        2.39 |
| Chunking  |        0.31 |
| Embedding |     3128.76 |
| Vector DB |        4.36 |
| **Total** | **3135.81** |

### Throughput

| Stage     | Items/sec |
| --------- | --------: |
| Chunking  | 22,428.89 |
| Embedding |      2.20 |
| Vector DB |  1,579.79 |

### Observation

Embedding accounts for approximately **99.8%** of the total pipeline execution time and is the primary optimization target.

---

# Experiment 001

## Objective

Evaluate whether increasing the Sentence Transformer batch size improves embedding throughput.

### Change

```python
DEFAULT_BATCH_SIZE = 64
```

↓

```python
DEFAULT_BATCH_SIZE = 128
```

---

## Result

| Metric         |       Batch 64 |      Batch 128 |
| -------------- | -------------: | -------------: |
| Embedding Time |    3128.76 sec |    4150.49 sec |
| Throughput     | 2.20 items/sec | 1.66 items/sec |
| Total Runtime  |    3135.81 sec |    4156.51 sec |

---

## Analysis

Increasing the batch size from **64** to **128** significantly reduced performance on the test hardware.

Observed impact:

* Embedding time increased by approximately **32.7%**
* Throughput decreased by approximately **24.5%**
* Total pipeline execution time increased significantly

The Intel Core i7-10510U CPU performs more efficiently using a batch size of **64** for this workload.

---

## Decision

**Rejected**

Reverted the batch size back to:

```python
DEFAULT_BATCH_SIZE = 64
```

---

# Current Best Configuration

| Parameter            | Value |
| -------------------- | ----- |
| Batch Size           | 64    |
| Normalize Embeddings | True  |
| Convert to NumPy     | True  |
| PyTorch Threads      | 4     |

---

# Future Experiments

* Profile SentenceTransformer `encode()` execution in greater detail.
* Investigate CPU inference optimizations.
* Evaluate newer embedding models while maintaining acceptable retrieval quality.
* Benchmark on GPU-enabled hardware for comparison.
* Compare embedding throughput across different CPUs.
