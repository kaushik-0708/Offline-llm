import streamlit as st
import chromadb
from sentence_transformers import SentenceTransformer
import ollama
import sqlite3
from datetime import datetime
from pypdf import PdfReader
from docx import Document
import os
import uuid
import pyperclip
import pytesseract
from io import BytesIO
from PIL import Image

try:
    import fitz
except ImportError:
    fitz = None

pytesseract.pytesseract.tesseract_cmd = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(BASE_DIR, "data")
CHROMA_DIR = os.path.join(BASE_DIR, "chroma_db")
CHAT_DB = os.path.join(BASE_DIR, "chat_history.db")

# ------------------ Page Config ------------------
st.set_page_config(
    page_title="Offline LLM",
    page_icon="🤖",
    layout="centered"
)

st.title("🤖 Offline LLM")
st.caption("Offline RAG chatbot with full document & chat management")

os.makedirs(DATA_DIR, exist_ok=True)

# ------------------ Utility: Chunking ------------------
def chunk_text(text, chunk_size=500, overlap=100):
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap

    return chunks

# ---------------- OCR FOR IMAGES ----------------
def ocr_image(image):
    return pytesseract.image_to_string(image)

def read_image(path):

    with Image.open(path) as img:

        text = ocr_image(img)

    return text

def extract_pdf_text(path):
    text_parts = []
    page_texts = []

    reader = PdfReader(path)

    for page in reader.pages:
        extracted = page.extract_text() or ""
        page_texts.append(extracted)

        if extracted.strip():
            text_parts.append(extracted)

    if fitz is None:
        return "\n".join(text_parts)

    pdf_doc = fitz.open(path)

    for page_index, page in enumerate(pdf_doc):
        normal_text = page_texts[page_index] if page_index < len(page_texts) else ""

        if not normal_text.strip():
            pixmap = page.get_pixmap(dpi=200)
            image = Image.open(BytesIO(pixmap.tobytes("png")))
            ocr_text = ocr_image(image)

            if ocr_text.strip():
                text_parts.append(ocr_text)

        for image_info in page.get_images(full=True):
            xref = image_info[0]
            image_data = pdf_doc.extract_image(xref)
            image_bytes = image_data.get("image")

            if not image_bytes:
                continue

            image = Image.open(BytesIO(image_bytes))
            ocr_text = ocr_image(image)

            if ocr_text.strip():
                text_parts.append(ocr_text)

    pdf_doc.close()

    return "\n".join(text_parts)

# ------------------ Chat History DB ------------------
def get_db_connection():
    return sqlite3.connect(CHAT_DB)

def init_db():
    conn = get_db_connection()
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS chat_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            role TEXT,
            message TEXT,
            timestamp TEXT
        )
    """)

    conn.commit()
    conn.close()

def save_message(role, message):
    conn = get_db_connection()
    cur = conn.cursor()

    cur.execute(
        "INSERT INTO chat_history (role, message, timestamp) VALUES (?, ?, ?)",
        (role, message, datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    )

    conn.commit()
    conn.close()

def load_chat_history():
    conn = get_db_connection()
    cur = conn.cursor()

    cur.execute("SELECT role, message FROM chat_history ORDER BY id")
    rows = cur.fetchall()

    conn.close()
    return rows

def clear_chat_history():
    conn = get_db_connection()
    cur = conn.cursor()

    cur.execute("DELETE FROM chat_history")

    conn.commit()
    conn.close()

init_db()

# ------------------ Load RAG Components ------------------
@st.cache_resource
def load_rag():
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")

    client = chromadb.PersistentClient(path=CHROMA_DIR)

    collection = client.get_or_create_collection(name="documents")

    return embed_model, collection

model, collection = load_rag()

def sync_collection_with_files(saved_files):
    saved_sources = set(saved_files)
    existing = collection.get(include=["metadatas"])
    stale_ids = []

    for item_id, metadata in zip(existing["ids"], existing["metadatas"]):
        source = metadata.get("source") if metadata else None
        if source not in saved_sources:
            stale_ids.append(item_id)

    if stale_ids:
        collection.delete(ids=stale_ids)

def is_copyable_answer(message):
    warning_messages = {
        "Please upload a document first.",
        "The document does not contain this information.",
    }
    message = message.strip()
    normalized_message = message.lower()

    if message in warning_messages:
        return False

    if "does not contain this information" in normalized_message:
        return False

    if "ollama is not running" in normalized_message:
        return False

    if message.startswith("⚠") or message.startswith("âš"):
        return False

    return True

def remove_sources_section(message):
    return message.split("\n\nSources:", 1)[0].strip()

# ------------------ Sidebar: Upload Documents ------------------
st.sidebar.header("📤 Upload Documents")

uploaded_files = st.sidebar.file_uploader(
    "Upload PDF / DOCX / TXT",
    type=["pdf", "docx", "txt", "png", "jpg", "jpeg"],
    accept_multiple_files=True
)

if uploaded_files:
    st.sidebar.info("Saving & embedding documents...")

    for uploaded_file in uploaded_files:

        save_path = os.path.join(DATA_DIR, uploaded_file.name)

        if os.path.exists(save_path):
            st.sidebar.warning(f"{uploaded_file.name} already exists")
            continue

        with open(save_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        text = ""

        if save_path.endswith(".txt"):
            text = open(
                save_path,
                "r",
                encoding="utf-8",
                errors="ignore"
            ).read()

        elif save_path.endswith(".pdf"):
            text = extract_pdf_text(save_path)

        elif save_path.endswith(".docx"):
            doc = Document(save_path)

            for p in doc.paragraphs:
                text += p.text + "\n"

        # IMAGE FILES
        elif save_path.endswith((".png", ".jpg", ".jpeg")):
                text = read_image(save_path)

        for chunk in chunk_text(text):

            collection.add(
                documents=[chunk],
                embeddings=[model.encode(chunk).tolist()],
                ids=[str(uuid.uuid4())],
                metadatas=[{"source": uploaded_file.name}]
            )

    st.sidebar.success("✅ Files uploaded and embedded")

# ------------------ Sidebar: Manage Documents ------------------
st.sidebar.header("🗂️ Manage Documents")

saved_files = os.listdir(DATA_DIR)
sync_collection_with_files(saved_files)

if saved_files:

    for file in saved_files:

        col1, col2 = st.sidebar.columns([3, 1])

        col1.write(file)

        if col2.button("❌", key=f"del_{file}"):

            os.remove(os.path.join(DATA_DIR, file))

            collection.delete(where={"source": file})

            st.sidebar.success(f"Deleted {file}")

            st.rerun()

else:
    st.sidebar.info("No documents available")

# ------------------ Sidebar: Clear Chat ------------------
st.sidebar.header("🧹 Chat Management")

if st.sidebar.button("Clear Chat History"):

    clear_chat_history()

    st.session_state.chat_history = []

    st.sidebar.success("Chat history cleared")

    st.rerun()

# ------------------ Session State ------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = load_chat_history()

# ------------------ Display Chat ------------------
for i, (role, msg) in enumerate(st.session_state.chat_history):

    with st.chat_message(role):

        display_msg = remove_sources_section(msg) if role == "assistant" else msg

        st.markdown(display_msg)

        # Copy button only for successful assistant answers
        if role == "assistant" and is_copyable_answer(display_msg):

            if st.button("📋 Copy", key=f"copy_{i}"):

                pyperclip.copy(display_msg)

                st.toast("Copied to clipboard!")

# ------------------ Chat Input ------------------
user_input = st.chat_input("Ask a question...")

if user_input:

    st.chat_message("user").markdown(user_input)

    st.session_state.chat_history.append(("user", user_input))

    save_message("user", user_input)

    saved_files = os.listdir(DATA_DIR)
    sync_collection_with_files(saved_files)

    if not saved_files or collection.count() == 0:

        answer = "Please upload a document first."

        with st.chat_message("assistant"):

            st.markdown(answer)

        st.session_state.chat_history.append(("assistant", answer))

        save_message("assistant", answer)

        st.stop()

    query_embedding = model.encode(user_input).tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=3,
        include=["documents"]
    )

    docs = results["documents"][0]

    if not docs:

        answer = "The document does not contain this information."

    else:

        context = "\n\n".join(docs)
        question_keywords = set(user_input.lower().split())
        context_keywords = set(context.lower().split())

        if len(question_keywords.intersection(context_keywords)) == 0:

            answer = "The document does not contain this information."

            with st.chat_message("assistant"):

                st.markdown(answer)

            st.session_state.chat_history.append(("assistant", answer))

            save_message("assistant", answer)

            st.stop()

        prompt = f"""
Use ONLY the context below to answer.
If not present, say:
"The document does not contain this information."

Context:
{context}

Question:
{user_input}

Answer:
"""

        try:

            response = ollama.chat(
                model="phi3:mini",
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            )

            answer = response["message"]["content"]

        except:

            answer = "⚠️ Ollama is not running."

    # ------------------ Assistant Message ------------------
    with st.chat_message("assistant"):

        st.markdown(answer)

        if is_copyable_answer(answer) and st.button(
            "📋 Copy",
            key=f"copy_new_{len(st.session_state.chat_history)}"
        ):

            pyperclip.copy(answer)

            st.toast("Copied to clipboard!")

    st.session_state.chat_history.append(("assistant", answer))

    save_message("assistant", answer)
