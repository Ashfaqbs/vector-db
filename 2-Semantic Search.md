## Section 2 Semantic Search

### 2.1 Intro

Traditional search engines look for the exact words you typed. This works for command-F on a page, but fails when “car” ≈ “automobile” or when the best answer is phrased differently from your query. **Semantic search** solves that gap: it retrieves items whose *meaning* parallels the query, not just the letters.

---

### 2.2 What (Definition & Key Pieces)

| Facet                   | Explanation                                                                             | Analogy                                                                          |
| ----------------------- | --------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------- |
| **Goal**                | Retrieve items whose *semantics*—the concepts expressed—lie closest to the query.       | *Asking a friend: “Songs that feel like Lo-fi beats” instead of naming artists.* |
| **Core data**           | Embeddings of queries and documents stored in a vector DB.                              | *Indexing books by “vibe coordinates” rather than page numbers.*                 |
| **Similarity metric**   | Cosine distance, dot-product, or Euclidean distance; lower distance ⇒ higher relevance. | *Angle between arrows: smaller angle = pointing in almost the same direction.*   |
| **Candidate set size**  | Top-k nearest neighbors (e.g., k = 20) returned in single-digit ms.                     | *Scanning only the nearest bookshelves, not the whole library.*                  |
| **Optional re-ranking** | A slower cross-encoder or LLM scores the top-k for final order.                         | *Pulling shortlisted books to a reading desk for closer inspection.*             |

---

### 2.3 Why (Motivation)

| Pain without semantics                                         | Improvement with semantic search                       |
| -------------------------------------------------------------- | ------------------------------------------------------ |
| Synonyms, paraphrases, misspellings miss.                      | Distance in embedding space naturally groups them.     |
| Sparse documents (tweets, code comments) lack keyword overlap. | Dense vectors amplify latent meaning.                  |
| Multilingual corpora need bridging terms.                      | Shared embedding space aligns “cat”, “gato”, “बिल्ली”. |
| Long queries: bag-of-words scoring dilutes intent.             | Whole-sentence embedding captures gist in one vector.  |

*Analogy*: You could search a grocery store aisle by reading every label (**keyword search**) or by using your sense of shape-and-color to spot cereal boxes even if branding changed (**semantic search**).

---

### 2.4 How (End-to-End Mechanics)

1. **Query encoding**
   The text “reset my kubernetes cluster” passes through the same embedding model used for the corpus → 768-dim vector **q**.

2. **Candidate retrieval**
   *Vector DB* (e.g., Qdrant, Pinecone) performs **ANN** (Approximate Nearest Neighbor) search:

   ```
   top_k = index.search(query_vector=q, limit=20)
   ```

   Internally, an HNSW graph walks from high-level hubs down to local nodes in O(log N) steps.

3. **Filtering & hybrid logic** *(optional)*
   · Metadata filters: `cluster="prod"` AND `namespace="sales"`
   · Boolean keyword clauses combined with vector similarity (“hybrid search”).

4. **Re-ranking** *(optional but common)*
   A cross-encoder BERT scores (query, doc) pairs for precision; or an LLM generates answer snippets and selects supporting docs.

5. **Response**
   Final list or snippet shown to user; downstream RAG pipelines may feed those docs into an LLM to craft an answer.

---

### 2.5 Variants & Design Patterns

| Pattern                        | Where it helps                                                   | Sketch                                                     |
| ------------------------------ | ---------------------------------------------------------------- | ---------------------------------------------------------- |
| **BM25 + ANN hybrid**          | Enterprise corpora with strict keyword filters (dates, IDs).     | Run BM25 filter → feed surviving docs into ANN search.     |
| **Late-interaction (ColBERT)** | Large passages where term-level granularity beats single-vector. | Store per-token embeddings, score max-sim per token.       |
| **Streaming semantic search**  | Real-time logs/events.                                           | Incrementally upsert vectors; HNSW handles dynamic graphs. |

---

### 2.6 Pitfalls & Mitigations

| Issue                      | Symptom                                         | Mitigation                                                        |
| -------------------------- | ----------------------------------------------- | ----------------------------------------------------------------- |
| **Embedding/domain drift** | New jargon (“EdgeDB”) retrieves poorly.         | Periodic re-training; adapters.                                   |
| **False positives**        | Conceptually close but contextually wrong docs. | Add metadata filters or cross-encoder re-ranker.                  |
| **Latency spikes**         | Index rebuilds or disk cold-starts.             | Warm-up queries; pin index in RAM; shard.                         |
| **Recall vs cost**         | Tight quantization drops recall.                | Tune codebook size; keep float32 for head index, int8 for leaves. |

---

### 2.7 Key Take-aways

1. **Semantic search compares ideas, not strings.**
2. Embeddings + a vector DB yield millisecond-level nearest-neighbor retrieval.
3. Hybrid keyword/vector and re-ranking layers sharpen precision.
4. Watch out for domain drift, quantization trade-offs, and cold-start latency.

---