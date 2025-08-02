from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer
import warnings

# Ignore Hugging Face warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# Connect to local Qdrant
client = QdrantClient(url="http://localhost:6333")

# Ensure fresh collection
COLLECTION = "quick_demo"
if client.collection_exists(COLLECTION):
    client.delete_collection(COLLECTION)

client.create_collection(
    collection_name=COLLECTION,
    vectors_config=models.VectorParams(size=384, distance=models.Distance.COSINE)
)

# Load model + encode
model = SentenceTransformer("all-MiniLM-L6-v2")
texts = [
    "What is a vector database?",
    "How do I reset my Kubernetes cluster?",
    "Python tips for web scraping"
]
vecs = model.encode(texts, normalize_embeddings=True)

# Insert vectors
client.upsert(
    collection_name=COLLECTION,
    points=[
        models.PointStruct(id=i+1, vector=vecs[i].tolist(), payload={"text": texts[i]})
        for i in range(len(texts))
    ]
)

# Run a semantic search
query = "Tell me about vector DBs"
query_vec = model.encode(query, normalize_embeddings=True)

results = client.search(
    collection_name=COLLECTION,
    query_vector=query_vec.tolist(),
    limit=2
)

print("\nTop matches:")
for hit in results:
    print(f"→ {hit.payload['text']} (score={hit.score:.3f})")



# OP
# (.venv) PS C:\tmp\git\vector-db\code> python Cilent-eg1.py
# C:\tmp\git\vector-db\code\Cilent-eg1.py:43: DeprecationWarning: `search` method is deprecated and will be removed in the future. Use `query_points` instead.
#   results = client.search(

# Top matches:
# → What is a vector database? (score=0.863)
# → How do I reset my Kubernetes cluster? (score=0.069)
# (.venv) PS C:\tmp\git\vector-db\code> 




#  Working 
# We encoded text into an embedding (vector) -> model.encode("Tell me about vector DBs", normalize_embeddings=True)
# The model (MiniLM-L6-v2) converted the sentence into a 384-dimensional vector like -> [-0.01, 0.47, -0.33, ...]  # total 384 numbers
# These numbers capture the semantic meaning of our sentence — not the exact words — but “what it’s about”.
# That vector was sent to Qdrant (our vector DB)
# Qdrant received your vector and used a nearest-neighbor index (HNSW) to answer:
# “What are the top 2 vectors in the collection that are closest in meaning to this one?”
# This is called  semantic search.


#  Qdrant calculated cosine similarity between our query and stored vectors
# We had stored 3 texts earlier, like:

# "What is a vector database?"

# "How do I reset my Kubernetes cluster?"

# "Python tips for web scraping"

# Each of those was also embedded into a vector during upsert and stored in the DB.

# Qdrant measured cosine similarity between:
    
# ```
# query_vec  ← ("Tell me about vector DBs")
#     and
# each stored_vec  ← (the 3 docs you upserted)

# ```

# What are these scores?
# From our output:

# → What is a vector database? (score=0.863)
# → How do I reset my Kubernetes cluster? (score=0.069)

# These are cosine similarity scores. Cosine similarity is:

# cosine(vec1, vec2) ≈ 1 → very similar
# cosine(vec1, vec2) ≈ 0 → unrelated
# cosine(vec1, vec2) < 0 → opposite (rare in normalized embeddings)

# So:

# 0.863 = strong semantic match

# 0.069 = weak match, likely a fallback

# (the third one "web scraping" likely had a lower score and didn't make top 2)
 
 
#  What is the vector DB doing?
#  | Step   | Action                            | Analogy                                                  |
# | ------ | --------------------------------- | -------------------------------------------------------- |
# | Encode | Text → vector                     | GPS location of a sentence in "meaning space"            |
# | Store  | Save that vector in DB            | Placing it on a shelf labeled by ID                      |
# | Search | Compare new vector to stored ones | Finding which shelf is nearest                           |
# | Score  | Return similarity (cosine)        | Angle between two arrows — closer = more aligned meaning |
