# 📖 RAG QA System (Book-based with Summarization)

This is a **Retrieval-Augmented Generation (RAG) QA system** for books, allowing users to upload PDFs, process them into text chunks, and ask questions that are answered using a combination of vector search and summarization.

The system leverages:
- **LangChain** for text splitting and vector storage.
- **FAISS** for similarity search.
- **HuggingFace Embeddings** (`sentence-transformers/all-MiniLM-L6-v2`) for vectorization.
- **Transformers** (`facebook/bart-large-cnn`) for summarization.
- **Streamlit** for a web-based user interface.

---

## ⚙️ Setup & Installation Instructions

1. **Clone the repository:**
```bash
git clone https://github.com/ShivangRustagi04/chatpdf
cd chatpdf
```

2. **Download the required libraries:**
```bash

pip install -r requirements.txt

```
3. **Run the app locally:**
```bash
streamlit run app.py
```


## ⚠️ Limitations & Security Compliance

File Size: Large PDF files (>50MB) may slow down processing or exceed memory limits.

Text Extraction: PDFs with scanned images or complex formatting may result in incomplete text extraction.

Summarization Limits: The summarizer may truncate or generalize context; verify with original content if precision is critical.

Security: Uploaded PDFs are stored only temporarily for processing. Ensure sensitive documents are handled carefully.

Offline Use: The app is designed for local/private deployment; no cloud storage or external API calls are made by default.