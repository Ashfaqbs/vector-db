## Section 4 Internal Index Algorithms

### 4.1 Intro

A vector DB’s magic lives in its **approximate-nearest-neighbor (ANN) index**.
Three work-horse algorithms dominate production in 2025:

1. **HNSW** – graph-based, high recall, fast inserts.
2. **IVF-PQ** – cluster + compress for memory-tight, billion-scale corpora.
3. **ScaNN** – Google’s hybrid framework marrying tree-pruning with learned quantization.

Think of them as three different **road maps**: highways (HNSW), city grids with zip-codes (IVF-PQ), and a sat-nav that combines both plus real-time traffic (ScaNN).

---

### 4.2 HNSW (Hierarchical Navigable Small-World)

|                |                                                                                                                                                                                                                                        |
| -------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **What**       | Builds a multi-layer graph where each node (vector) connects to its neighbors; upper layers are sparse “express lanes”, lower layers dense local roads.                                                                                |
| **Why**        | Offers logarithmic search complexity and supports incremental upserts without global rebuilds.                                                                                                        |
| **How**        | **Insertion**: new point starts at top layer, descends greedily, linking to nearest nodes at each level.<br>**Query**: start from an entry point on the top layer, walk down while shrinking the search radius until k nearest remain. |
| **Analogy**    | **Elevators + corridors** in a skyscraper: express elevators move you floor-to-floor (layers), then corridors guide you to the exact office (vector).                                                                                  |
| **Trade-offs** | • High recall at modest RAM.<br>• Writes slightly slower than IVF lists but still real-time.<br>• Entire graph should fit in memory for best latency.                                                                                  |

Recent engines (Qdrant, Pinecone, Milvus) default to HNSW because it balances recall and latency with minimal tuning.

---

### 4.3 IVF-PQ (Inverted File + Product Quantization)

|                |                                                                                                                                                                                                                         |
| -------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **What**       | Two-step strategy:<br>1) **IVF** partitions the vector space into *n* coarse centroids (like postal codes).<br>2) **PQ** compresses the residual (the difference between vector and its centroid) into short codewords. |
| **Why**        | Shrinks RAM 8–16× versus float32 while hitting sub-second build times on GPUs. Excellent for multi-billion vector archives where every byte matters.                                          |
| **How**        | **Search**: find the nearest centroids (usually 5–10 % of buckets), decompress only those codebooks, then scan compressed codes to rank candidates.                                                                     |
| **Analogy**    | **Warehouse + SKU codes**: first locate the right warehouse zone (IVF), then use a short SKU to fetch the exact item (PQ).                                                                                              |
| **Trade-offs** | • Lower RAM and fast bulk inserts.<br>• Slight recall drop from quantization; tune “n-probe” (how many centroids to scan) for quality/latency trade-off.                    |

GPU libraries such as NVIDIA cuVS accelerate IVF-PQ build/search pipelines.

---

### 4.4 ScaNN (Scalable Nearest Neighbors – Google)

|                |                                                                                                                                                                                         |
| -------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **What**       | Combines **partitioning** (e.g., k-means tree) to prune search space, **asymmetric quantization** to compress vectors, and **optimized SIMD kernels** for fast distance math.           |
| **Why**        | Achieves near-HNSW recall with lower memory than pure graphs and better CPU cache locality; shines in large-language-model serving at Google scale. |
| **How**        | Multi-stage:<br>1) Partition database.<br>2) Pre-compute quantized residuals.<br>3) At query time, quickly score partitions, then run a high-precision re-rank on a narrowed set.       |
| **Analogy**    | **Airport security**: coarse metal detector (partition filter) first, then detailed bag scan (re-rank) only for flagged passengers.                                                     |
| **Trade-offs** | • More complex to tune (must pick tree depth, quantizer bits).<br>• Insert path slower—best for mostly read-heavy workloads.                                                            |

Open-source ScaNN library integrates with TensorFlow and standalone Python.

---

### 4.5 Choosing an Index

| Need                                                 | Best fit               | Rationale                                           |
| ---------------------------------------------------- | ---------------------- | --------------------------------------------------- |
| **High QPS, streaming upserts**                      | HNSW                   | Handles real-time inserts with minimal recall loss. |
| **Memory-constrained, archive-scale (10B+ vectors)** | IVF-PQ                 | Aggressive compression keeps cost down.             |
| **CPU-only, hybrid latency/recall sweet spot**       | ScaNN                  | Partition + quantization balances both.             |
| **Edge device < 1 GB RAM**                           | PQ-only or binary hash | Extreme compression at acceptable recall.           |

---

### 4.6 Key Take-aways

1. **HNSW** = layered graph, excels at dynamic data and high recall.
2. **IVF-PQ** = cluster + codebook, trims RAM for trillion-token corpora.
3. **ScaNN** = hybrid pruning + quantization, efficient on CPU at Google-scale.
4. Pick the algorithm by matching **workload pattern (read/write), memory budget, recall target, and hardware**.
