import chromadb
from sentence_transformers import SentenceTransformer
import ollama

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Load persistent ChromaDB
client = chromadb.PersistentClient(path="../chroma_db")
collection = client.get_or_create_collection(name="documents")

print("\n⚪️ Offline LLM Started...")
print("Type your question and press Enter")

while True:
    # Ask user question
    question = input("👦 You: ").strip()

    if question.lower() in ["exit", "quit"]:
        print("\n👋 Exiting Offline LLM. Goodbye!")
        break

    # Convert question to embedding
    query_embedding = model.encode(question).tolist()

    # Retrieve top relevant chunks
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=3,
        include=["documents"]
    )

    documents = results["documents"][0]

    # 🚨 Guard 1: No retrieved content
    if not documents or all(len(doc.strip()) == 0 for doc in documents):
        print("\n🤖 LLM: The document does not contain this information.\n")
        continue

    # Combine context
    context = "\n\n".join(documents)

    # 🚨 Guard 2: Relevance check (prevents outside knowledge)
    question_keywords = set(question.lower().split())
    context_keywords = set(context.lower().split())

    if len(question_keywords.intersection(context_keywords)) == 0:
        print("\n🤖 LLM: The document does not contain this information.\n")
        continue

    # Build strict RAG prompt
    prompt = f"""
You are a strict document-based assistant.

Rules:
- Use ONLY the provided context.
- DO NOT use any external knowledge.
- If the answer is not present in the context, say:
  "The document does not contain this information."

Context:
{context}

Question:
{question}

Answer:
"""

    # Call Ollama
    response = ollama.chat(
        model="phi3:mini",
        messages=[
            {"role": "user", "content": prompt}
        ]
    )

    print("\n🤖 LLM:")
    print(response["message"]["content"])
    print()
