from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue
)

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")  # 384-dim embeddings

# Connect to local Qdrant
client = QdrantClient("http://localhost:6333")

# === Create a fresh collection ===
COLLECTION = "hybrid_demo"
if client.collection_exists(COLLECTION):
    client.delete_collection(COLLECTION)

client.create_collection(
    collection_name=COLLECTION,
    vectors_config=VectorParams(size=384, distance=Distance.COSINE)
)

# === Sample text + metadata ===
entries = [
    ("Kubernetes cluster troubleshooting guide", "guide", "en"),
    ("Restart services on k8s node", "guide", "en"),
    ("French Kubernetes admin steps", "guide", "fr"),
    ("Web scraping with Python", "tutorial", "en"),
    ("Beginner's guide to NLP", "guide", "en"),
    ("Kubernetes reset checklist", "cheatsheet", "en"),
    ("Docker basics in Spanish", "guide", "es")
]

# === Embed + insert ===
texts = [e[0] for e in entries]
vectors = model.encode(texts, normalize_embeddings=True)

points = [
    PointStruct(
        id=i + 1,
        vector=vectors[i].tolist(),
        payload={
            "text": entries[i][0],
            "type": entries[i][1],
            "lang": entries[i][2]
        }
    )
    for i in range(len(entries))
]

client.upsert(collection_name=COLLECTION, points=points)

# === Search with hybrid filter: type = "guide", lang = "en" ===
query = "how to fix Kubernetes problems"
query_vector = model.encode(query, normalize_embeddings=True)

results = client.search(
    collection_name=COLLECTION,
    query_vector=query_vector.tolist(),
    limit=3,
    query_filter=Filter(
        must=[
            FieldCondition(key="type", match=MatchValue(value="guide")),
            FieldCondition(key="lang", match=MatchValue(value="en"))
        ]
    )
)

# === Print results ===
print("\n🔎 Hybrid search results (type='guide', lang='en'):")
for hit in results:
    print(f"→ ID={hit.id}, score={hit.score:.3f}")
    print(f"   text: {hit.payload['text']}")
    print(f"   type: {hit.payload['type']}, lang: {hit.payload['lang']}\n")


# OP
# (.venv) PS C:\tmp\git\vector-db\code> python .\Hybrid-Search.py
# C:\tmp\git\vector-db\code\.venv\Lib\site-packages\torch\nn\modules\module.py:1762: FutureWarning: `encoder_attention_mask` is deprecated and will be removed in version 4.55.0 for `BertSdpaSelfAttention.forward`.
#   return forward_call(*args, **kwargs)
# C:\tmp\git\vector-db\code\Hybrid-Search.py:62: DeprecationWarning: `search` method is deprecated and will be removed in the future. Use `query_points` instead.
#   results = client.search(

# 🔎 Hybrid search results (type='guide', lang='en'):
# → ID=1, score=0.757
#    text: Kubernetes cluster troubleshooting guide
#    type: guide, lang: en

# → ID=2, score=0.512
#    text: Restart services on k8s node
#    type: guide, lang: en

# → ID=5, score=0.086
#    text: Beginner's guide to NLP
#    type: guide, lang: en

# (.venv) PS C:\tmp\git\vector-db\code> 


# These are filtered results:

# All have type: guide ✅

# All have lang: en ✅

# Then Qdrant returned the top 3 most semantically relevant to your query, ranked by cosine similarity (score near 1 = more similar)

# So:

# ID=1 is most relevant to “how to fix Kubernetes problems”

# ID=5 (NLP) is the least, but still part of the filtered set



# What We Did (Step-by-Step)

# | Step                                            | Action                                                    |
# | ----------------------------------------------- | --------------------------------------------------------- |
# | ✅ Created `hybrid_demo` collection              | With 384-dim vector space using cosine similarity         |
# | ✅ Embedded 7 text snippets                      | Using the MiniLM model into 384-dim semantic vectors      |
# | ✅ Tagged each vector with **payload**           | e.g., `"type": "guide"`, `"lang": "en"`                   |
# | ✅ Searched for a query                          | `"how to fix Kubernetes problems"`                        |
# | ✅ Applied **hybrid filter**                     | Only return docs where `type="guide"` **AND** `lang="en"` |
# | ✅ Got results ranked by **semantic similarity** | Only from the filtered subset                             |




#  Why It’s Useful
#  This is what real-world vector search needs:
     
# | Without filters                                                            | With filters                                                                   |
# | -------------------------------------------------------------------------- | ------------------------------------------------------------------------------ |
# | May return random Kubernetes answers in French, blogs, or unrelated guides | Only returns semantically relevant content that meets **your filter criteria** |
# | Can’t do tenant-based search or type-specific results                      | Can isolate by `user_id`, `doc_type`, `category`, `date`, etc.                 |
# | Gets noisy as our collection grows                                        | Keeps relevance high by slicing the candidate pool first                       |


# We controlled the context of search — that’s essential for precise answers, fast UI, and scoped security in RAG, support bots, or search apps.



# What Qdrant Did Internally
# Indexed all vectors in a HNSW graph

# Stored payloads alongside vectors

# When we queried:

# Filtered only points with type = guide and lang = en

# Searched those via cosine similarity

# Returned top-k by angle (cosine score)

# ⚙️ This filtering happens before search, so it’s fast and scalable.


# Real-world usage examples

# | App Type   | Example Hybrid Query                                                                                     |
# | ---------- | -------------------------------------------------------------------------------------------------------- |
# | LLM Q\&A   | Find docs semantically similar to “reset cluster”, where `type="internal_guide"` and `tenant_id="lowes"` |
# | Search bar | Search “lofi jazz” in a music app, but only where `genre="lofi"` and `language="instrumental"`           |
# | E-commerce | Similar to “Nike running shoes”, but filter `brand="Nike"`, `category="running"`, `price<300`            |

