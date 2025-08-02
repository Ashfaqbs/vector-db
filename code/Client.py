from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer

# pip install qdrant-client sentence-transformers


# connect
client = QdrantClient("http://localhost:6333")

# (re)create collection
client.recreate_collection(
    "demo",
    vectors_config=models.VectorParams(size=384, distance=models.Distance.COSINE)
)

# embed & UPSERT two points
# UPSERT Means
# Update or insert in one command.
# If the ID already exists, Qdrant updates the vector/payload in-place; if the ID is new, it inserts a fresh point.

model = SentenceTransformer("all-MiniLM-L6-v2")
for pid, text in [(1, "hello"), (2, "hi there")]:
    vec = model.encode(text, normalize_embeddings=True).tolist()
    client.upsert("demo", [models.PointStruct(id=pid, vector=vec)])

# run a search
hits = client.search("demo",
                     model.encode("greetings", normalize_embeddings=True).tolist(),
                     limit=1)
print(hits[0].id, hits[0].score)
