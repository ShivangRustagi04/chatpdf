import streamlit as st
import json
import pdfplumber
from transformers import pipeline
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# ---------------------------
# Step 1: Extract Text from PDF
# ---------------------------
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        with pdfplumber.open(pdf) as pdf_reader:
            for page_num, page in enumerate(pdf_reader.pages):
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
                else:
                    text += f"\n[Warning: No extractable text found on page {page_num+1}]\n"
    return text

# ---------------------------
# Step 2: Chunk Text
# ---------------------------
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    chunks = text_splitter.split_text(text)
    return chunks

# ---------------------------
# Step 3: Build Vector Store
# ---------------------------
def get_vector_store(text_chunks):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

    # Save chunks with metadata
    with open("book_chunks.json", "w") as f:
        json.dump([{"chunk": c} for c in text_chunks], f, indent=4)

# ---------------------------
# Step 4: Summarization Model
# ---------------------------
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

def user_input(user_question):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = db.similarity_search(user_question, k=3)

    # Combine retrieved chunks into one context
    context = "\n".join([d.page_content for d in docs])

    # Generate summarized answer
    try:
        summary = summarizer(context, max_length=150, min_length=30, do_sample=False)
        answer = summary[0]["summary_text"]
    except Exception:
        answer = "Answer not found in the book."

    # Display in UI
    st.subheader("Answer")
    st.write(answer)

    st.subheader("Relevant Book Chunks")
    for i, d in enumerate(docs):
        st.write(f"**Chunk {i+1}:** {d.page_content[:500]}...")

# ---------------------------
# Step 5: Streamlit App
# ---------------------------
def main():
    st.set_page_config("Book QA System")
    st.header("📖 RAG QA System (Book-based with Summarization)")

    user_question = st.text_input("Ask a question from the book:")

    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.title("Menu")
        pdf_docs = st.file_uploader("Upload the Book (PDF)", accept_multiple_files=True)
        if st.button("Submit & Process"):
            with st.spinner("Processing book..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("Book processed successfully!")

if __name__ == "__main__":
    main()
