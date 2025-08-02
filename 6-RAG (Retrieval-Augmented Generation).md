## Section 6 – Retrieval-Augmented Generation (RAG)

---

### 8.1 What is RAG?

Retrieval-Augmented Generation (RAG) is a method where a language model is provided with relevant external information at query time. The model uses this information to generate responses that are grounded in factual or domain-specific data. This technique is especially useful when the language model’s training data is incomplete, outdated, or not domain-aware.

---

### 8.2 Why use RAG?

| Challenge                                    | RAG Benefit                                           |
| -------------------------------------------- | ----------------------------------------------------- |
| Language models hallucinate or guess facts   | Retrieved context guides accurate generation          |
| Model doesn’t know private/internal info     | External data can be injected at runtime              |
| Costly to fine-tune models for new knowledge | Retrieval layer keeps knowledge external and editable |
| Need live or rapidly changing answers        | Vector DB content can be updated in real time         |

---

### 8.3 RAG Process Architecture

```text
          ┌────────────────────┐
          │  User question     │
          └────────┬───────────┘
                   ▼
       ┌───────────────────────┐
       │ Embed question vector │
       └────────┬──────────────┘
                ▼
       ┌───────────────────────────────┐
       │ Vector DB (e.g., Qdrant)      │
       │ Search top-k document chunks  │
       └────────┬──────────────────────┘
                ▼
       ┌──────────────────────────────┐
       │ Compose a prompt:            │
       │ “Answer using these docs…”   │
       └────────┬─────────────────────┘
                ▼
       ┌──────────────────────────────┐
       │ LLM (e.g., GPT, Gemini)      │
       │ Generates a factual answer   │
       └──────────────────────────────┘
```

---

### 8.4 Example Dataset

Document chunks can be short paragraphs or sentences from longer files. Each chunk is embedded and stored in a vector database, along with metadata.

Example input data:

```text
Doc title: "K8s Reset"
Chunk: "To reset your Kubernetes cluster, use `kubeadm reset`. This removes all Kubernetes components."
Type: "howto"

Doc title: "K8s Basics"
Chunk: "Kubernetes is an open-source system for automating deployment, scaling, and management of containerized applications."
Type: "reference"
```

---

### 8.5 Embedding and Storing Documents

A vector database such as Qdrant is used to store document embeddings along with metadata.

```python
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct

client = QdrantClient("http://localhost:6333")
model = SentenceTransformer("all-MiniLM-L6-v2")

client.create_collection(
    collection_name="doc_demo",
    vectors_config=VectorParams(size=384, distance=Distance.COSINE)
)

texts = [
    ("K8s Reset", "To reset your Kubernetes cluster, use `kubeadm reset`. This removes all Kubernetes components.", "howto"),
    ("K8s Basics", "Kubernetes is an open-source system for automating deployment, scaling, and management of containerized applications.", "reference")
]

vectors = model.encode([t[1] for t in texts], normalize_embeddings=True)

points = [
    PointStruct(
        id=i + 1,
        vector=vectors[i].tolist(),
        payload={"doc_title": texts[i][0], "chunk": texts[i][1], "type": texts[i][2]}
    )
    for i in range(len(texts))
]

client.upsert(collection_name="doc_demo", points=points)
```

---

### 8.6 Retrieval Query with Filter

To simulate a RAG query, the system embeds the incoming question and performs a filtered vector search to get top-k context chunks.

```python
from qdrant_client.models import Filter, FieldCondition, MatchValue

query = "How to reset Kubernetes cluster"
query_vec = model.encode(query, normalize_embeddings=True)

results = client.search(
    collection_name="doc_demo",
    query_vector=query_vec.tolist(),
    limit=2,
    query_filter=Filter(
        must=[FieldCondition(key="type", match=MatchValue(value="howto"))]
    )
)
```

---

### 8.7 Prompt Composition for LLM

The retrieved chunks are merged into a system prompt for the language model.

```python
retrieved_chunks = [hit.payload["chunk"] for hit in results]

prompt = f"""You are an AI assistant helping users with Kubernetes tasks.
Use only the information below to answer the question.

Context:
{"".join(f"- {chunk}\n" for chunk in retrieved_chunks)}

Question:
How to reset Kubernetes cluster?

Answer:"""

print(prompt)
```

---

### 8.8 Example Prompt Output

```
You are an AI assistant helping users with Kubernetes tasks.
Use only the information below to answer the question.

Context:
- To reset your Kubernetes cluster, use `kubeadm reset`. This removes all Kubernetes components.

Question:
How to reset Kubernetes cluster?

Answer:
```

---

### 8.9 Key Takeaways

* RAG enables language models to generate accurate, context-aware answers using dynamic external documents.
* The vector DB acts as a retrieval layer, selecting the most relevant content based on semantic similarity and filters.
* Document chunks can be filtered by metadata such as `type`, `user`, `source`, or `timestamp`.
* Prompt composition is the step where retrieved chunks are included in the input for the model.
