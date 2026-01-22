from qdrant_client import QdrantClient, models

client = QdrantClient(url="http://localhost:6333")

client.recreate_collection(
    collection_name="conversation_memory",
    vectors_config=models.VectorParams(
        size=1024,
        distance=models.Distance.COSINE
    )
)

print("Memory collection created")