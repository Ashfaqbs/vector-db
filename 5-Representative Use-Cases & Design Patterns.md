## Section 5 Representative Use-Cases & Design Patterns

### 5.1 Intro

Embeddings + a vector database unlock workflows that keyword systems cannot satisfy. This section groups real-world scenarios into repeatable patterns so you can choose an architecture rather than start from scratch.

---

### 5.2 What (Canonical Use-Cases)

| #       | Domain                                   | Problem Statement                                                              | Analogy                                                              |
| ------- | ---------------------------------------- | ------------------------------------------------------------------------------ | -------------------------------------------------------------------- |
| **U-1** | **Retrieval-Augmented Generation (RAG)** | Feed an LLM with semantically closest chunks to ground its answer.             | *Consulting subject-matter books before writing an essay.*           |
| **U-2** | **Personalized Recommendation**          | Suggest items (products, videos) whose vectors lie near a user profile vector. | *Spotify’s “Discover Weekly” built from similar sound fingerprints.* |
| **U-3** | **Image / Audio Similarity**             | Find pictures or audio snippets that *feel* like a query sample.               | *Shazam for images.*                                                 |
| **U-4** | **Anomaly / Fraud Detection**            | Flag vectors far from any cluster centroid as outliers.                        | *Airport security noticing a bag that looks unlike the rest.*        |
| **U-5** | **Cross-lingual & Multimodal Search**    | English query retrieves French docs or images, all embedded into one space.    | *Universal remote that works with every TV brand.*                   |
| **U-6** | **Code Intelligence**                    | Autocomplete and “find similar function” in large repos via code embeddings.   | *Stack Overflow suggestions baked into IDE.*                         |

---

### 5.3 Why (Benefits Across Scenarios)

| Pain Point                                         | Vector Solution                                 |
| -------------------------------------------------- | ----------------------------------------------- |
| Vocabulary mismatch, synonyms, slang               | Geometry groups meanings automatically.         |
| Cold-start items with no user clicks               | Content embeddings provide immediate neighbors. |
| Multimodal fusion is brittle with separate indexes | Joint embedding space unifies modal boundaries. |
| Sparse data (short tweets, log lines)              | Dense vectors amplify latent semantics.         |

---

### 5.4 How (End-to-End Pattern per Use-Case)

#### Pattern A **LLM + RAG**

1. **Chunk & embed** domain documents → store in VDB.
2. **Query** comes in → embed → ANN search (k ≈ 20).
3. **Re-rank** top-k via cross-encoder or LLM scoring.
4. **Compose prompt**: `system + context + question`.
5. **Generate answer**; optionally write new embeddings of the answer back for future retrieval.

*Design tip*: keep chunk size 200-400 tokens; larger chunks hurt recall, smaller chunks raise storage cost.

---

#### Pattern B **Realtime Recommendations**

1. **User activity** updates a running average vector (profile).
2. On page load, query VDB for top-n items near profile vector, filtered by metadata (in-stock, region).
3. Log interactions and periodically retrain embedding model or profile aggregator.

*Latency budget*: < 50 ms total ⇒ use HNSW in-RAM and pre-computed profile vectors.

---

#### Pattern C **Anomaly Detection**

1. Maintain centroid and variance per cluster (k-means or HDBSCAN).
2. New vector arrives → distance > τ? → flag event.
3. Optionally store flagged vectors in a separate “quarantine” collection for manual review.

*Threshold selection*: derive τ from 99th-percentile intra-cluster distance on validation data.

---

### 5.5 Design Templates

| Template                 | Read/Write Mix                   | Index Choice            | Quantization   | Infra Sketch             |
| ------------------------ | -------------------------------- | ----------------------- | -------------- | ------------------------ |
| **Edge RAG**             | 80 % read, light writes          | HNSW                    | FP16 or Int8   | Alpine + CPU, 1–2 GB RAM |
| **Billion-item Catalog** | Heavy read, batch writes nightly | IVF-PQ (nprobe = 8)     | PQ 8×          | 4 × A100 GPUs + SSD      |
| **Realtime Fraud**       | Continuous writes, low reads     | HNSW + periodic rebuild | None (float32) | Kafka → Flink → VDB      |

---

### 5.6 Take-aways

1. **Map need to pattern**: RAG, recommendations, similarity, anomaly.
2. **Latency dictates index**: HNSW for < 20 ms, IVF-PQ for RAM savings, ScaNN for CPU-only fleets.
3. **Hybrid filters matter**: Always combine vector scores with metadata when domain allows.
4. **Iterate**: Monitor recall & latency; adjust dimension, quantization, and k/nprobe rather than rewriting pipelines.
