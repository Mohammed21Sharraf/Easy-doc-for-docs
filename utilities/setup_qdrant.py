from qdrant_client import QdrantClient, models

def setup_qdrant():
    client = QdrantClient("http://localhost:6333")
    return client

def create_collection(client, collenction_name, size):
    if not client.collection_exists(collenction_name):
        client.create_collection(
            collection_name=collenction_name,
            vectors_config=models.VectorParams(size=size, distance=models.Distance.COSINE)  
        )

def upload_collection(client, collection_name, embeddings, metadata):
    client.upload_collection(
        collection_name=collection_name,
        vectors=embeddings,
        payload=metadata,
        ids=None
    )
