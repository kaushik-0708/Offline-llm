import os
import chromadb
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader
from docx import Document

DATA_DIR = "../data"
CHROMA_DIR = "../chroma_db"

def chunk_text(text, chunk_size=500, overlap=100):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap
    return chunks

print("Loading embedding model...")
model = SentenceTransformer("all-MiniLM-L6-v2")

client = chromadb.PersistentClient(path=CHROMA_DIR)
collection = client.get_or_create_collection(name="documents")

doc_id = 0
print("Embedding documents...")

for filename in os.listdir(DATA_DIR):
    file_path = os.path.join(DATA_DIR, filename)
    text = ""

    # TXT FILE
    if filename.endswith(".txt"):
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()

    # PDF FILE
    elif filename.endswith(".pdf"):
        reader = PdfReader(file_path)
        for page in reader.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted

    # DOCX FILE
    elif filename.endswith(".docx"):
        doc = Document(file_path)
        for para in doc.paragraphs:
            text += para.text + "\n"

    else:
        continue  # unsupported file

    if not text.strip():
        print(f"⚠️ Skipping empty file: {filename}")
        continue

    chunks = chunk_text(text)

    print(f"\nFile: {filename}")
    print(f"Chunks created: {len(chunks)}")

    for chunk in chunks:
        embedding = model.encode(chunk).tolist()
        collection.add(
            documents=[chunk],
            embeddings=[embedding],
            ids=[f"doc_{doc_id}"]
        )
        doc_id += 1

print("✅ PDF, DOCX, and TXT embeddings stored successfully")
