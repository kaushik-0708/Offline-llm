# 🤖 Offline OCR-Enhanced LLM

An AI-powered Offline OCR-Enhanced Document Question Answering System built using Streamlit, ChromaDB, Ollama, SentenceTransformers, and OCR technologies.

The system allows users to upload documents and ask questions from them completely offline using a local Large Language Model (LLM).

---

# 🚀 Features

✅ Fully Offline AI System  
✅ Retrieval-Augmented Generation (RAG)  
✅ Semantic Search using Vector Embeddings  
✅ OCR Support for Image-Based PDFs  
✅ Supports PDF, DOCX, TXT, PNG, JPG, JPEG  
✅ Scanned PDF Text Extraction  
✅ Local LLM using Ollama  
✅ Chat History Management  
✅ ChromaDB Vector Storage  
✅ Interactive Streamlit UI  
✅ Privacy-Focused Local Processing  

---

# 🧠 Technologies Used

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| Streamlit | Web UI |
| Ollama | Local LLM execution |
| Phi-3 Mini | Local language model |
| SentenceTransformers | Text embeddings |
| ChromaDB | Vector database |
| SQLite | Chat history storage |
| PyPDF | PDF text extraction |
| python-docx | DOCX processing |
| pytesseract | OCR extraction |
| Pillow | Image processing |
| pdf2image | Convert scanned PDFs to images |
| PyMuPDF | Extract images from PDFs |

---

# 📌 Project Objective

The objective of this project is to build a fully offline intelligent document assistant capable of:

- Understanding uploaded documents
- Extracting text from scanned PDFs and images
- Performing semantic retrieval
- Generating context-aware answers using local LLMs
- Maintaining user privacy by avoiding cloud APIs

---

# ⚙️ System Workflow

```text
User Uploads Document
        ↓
Text Extraction / OCR
        ↓
Chunking
        ↓
Embedding Generation
        ↓
ChromaDB Vector Storage
        ↓
Semantic Retrieval
        ↓
Ollama Local LLM
        ↓
Generated Answer
```

---

# 🏗️ Project Architecture

```text
PDF / DOCX / TXT / Images
            ↓
   Text Extraction + OCR
            ↓
        Chunking
            ↓
 SentenceTransformer Embeddings
            ↓
        ChromaDB
            ↓
      Semantic Search
            ↓
      Ollama (Phi-3)
            ↓
       Final Answer
```

---

# 📂 Supported File Types

- PDF
- DOCX
- TXT
- PNG
- JPG
- JPEG

---

# 🔍 OCR Support

This project supports OCR (Optical Character Recognition) using Tesseract OCR.

OCR allows the system to:
- Read scanned PDFs
- Extract text from images
- Understand screenshots and diagrams containing text

---

# 📁 Project Structure

```text
offline_llm/
│
├── app/
│   ├── ui.py
│   ├── embed.py
│   ├── rag_answer.py
│   ├── search.py
│   └── test.py
│
├── chroma_db/
├── data/
├── venv/
├── chat_history.db
├── requirements.txt
└── README.md
```

---

# ⚡ Installation

## 1️⃣ Clone Repository

```bash
git clone https://github.com/your-username/offline_llm.git
cd offline_llm
```

---

## 2️⃣ Create Virtual Environment

```bash
python -m venv venv
```

---

## 3️⃣ Activate Virtual Environment

### Windows

```bash
venv\Scripts\activate
```

### Linux / Mac

```bash
source venv/bin/activate
```

---

## 4️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

# 🧾 Install Tesseract OCR

Download and install Tesseract OCR:

https://github.com/UB-Mannheim/tesseract/wiki

After installation, verify:

```bash
tesseract --version
```

---

# 🤖 Install Ollama

Download Ollama:

https://ollama.com/download

Pull model:

```bash
ollama pull phi3:mini
```

---

# ▶️ Run the Project

Start Ollama model:

```bash
ollama run phi3:mini
```

Run Streamlit app:

```bash
streamlit run app/ui.py
```

---

# 💬 Example Questions

- What technologies are used?
- Summarize the document
- What is the project objective?
- Explain the architecture diagram
- What database is used?

---

# 📊 Key Features of RAG Pipeline

- Semantic similarity search
- Vector embeddings
- Context-aware answering
- Offline AI inference
- OCR-enhanced document understanding

---

# 🔐 Advantages

✅ Completely Offline  
✅ Privacy Focused  
✅ No Internet Required  
✅ Supports OCR Documents  
✅ Fast Retrieval  
✅ Lightweight Architecture  

---

# ⚠️ Limitations

- OCR accuracy depends on image quality
- Small LLM may produce limited reasoning
- Complex diagrams are not fully visually understood

---

# 🔮 Future Improvements

- Voice input/output
- Better OCR preprocessing
- Multi-language support
- Advanced retrieval techniques
- Larger local LLMs
- GPU acceleration

---

# 🎓 Applications

- Educational document assistant
- Research paper analysis
- Offline AI chatbot
- Enterprise document QA
- Personal knowledge management

---

# 👨‍💻 Author

Kaushik Mane
(Artificial Intelligence & Data Science Engineer)

---

# 📜 License

This project is developed for educational and research purposes.
