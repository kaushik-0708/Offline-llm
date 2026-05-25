import chromadb
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

# ✅ IMPORTANT: Same PersistentClient
client = chromadb.PersistentClient(path="../chroma_db")

collection = client.get_or_create_collection(name="documents")

count = collection.count()
print(f"📊 Total vectors in database: {count}")

query = input("Enter your question: ")

if count == 0:
    print("❌ No embeddings found.")
    exit()

query_embedding = model.encode(query).tolist()

results = collection.query(
    query_embeddings=[query_embedding],
    n_results=1,
    include=["documents"]
)

print("\nRESULT:")
print(results["documents"][0][0])
