import streamlit as st
import json
import pdfplumber
import re
from transformers import pipeline
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

def get_pdf_text(pdf_docs):
    text = ""
    if not pdf_docs:
        st.warning("No PDF uploaded.")
        return text
    for pdf in pdf_docs:
        try:
            with pdfplumber.open(pdf) as pdf_reader:
                for page_num, page in enumerate(pdf_reader.pages):
                    page_text = page.extract_text()
                    if page_text:
                        cleaned_text = re.sub(r'\s+', ' ', page_text)
                        cleaned_text = re.sub(r'[^\x00-\x7F]+', '', cleaned_text)
                        text += cleaned_text + "\n"
                    else:
                        text += f"\n[Warning: No extractable text on page {page_num+1}]\n"
        except Exception as e:
            st.error(f"Failed to process {pdf.name}: {e}")
    return text

def get_text_chunks(text):
    if not text.strip():
        st.warning("No text found to split into chunks.")
        return []
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    if not text_chunks:
        st.error("Cannot create FAISS index: no text chunks available.")
        return None
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")
    with open("book_chunks.json", "w") as f:
        json.dump([{"chunk": c} for c in text_chunks], f, indent=4)
    st.success("FAISS index and chunks saved successfully!")
    return vector_store

@st.cache_resource
def load_summarizer():
    return pipeline("summarization", model="facebook/bart-large-cnn")

summarizer = load_summarizer()

def is_question_relevant(question, context):
    """
    Check if the retrieved context is actually relevant to the question.
    This is a simple implementation that could be enhanced with more sophisticated NLP.
    """
    question_lower = question.lower()
    context_lower = context.lower()
    
    # Check if key question words appear in the context
    question_words = set(question_lower.split())
    context_words = set(context_lower.split())
    
    # Count how many question words appear in the context
    matching_words = question_words.intersection(context_words)
    
    # If less than 30% of question words appear in context, consider it irrelevant
    if len(question_words) > 0 and len(matching_words) / len(question_words) < 0.3:
        return False
    
    return True

def user_input(user_question):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    try:
        db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    except Exception:
        st.error("FAISS index not found. Please process a book first.")
        return

    docs = db.similarity_search(user_question, k=3)
    if not docs:
        st.info("No relevant content found in the book for this question.")
        return

    # Collect pages for final answer
    pages_used = sorted(set([d.metadata.get("page", "N/A") for d in docs]))

    # Create context from retrieved docs
    context = "\n".join([d.page_content for d in docs])
    
    # Check if the retrieved content is actually relevant to the question
    if not is_question_relevant(user_question, context):
        st.info("I'm sorry, but this question doesn't seem to be covered in the book content. Please ask a question related to the PDF content.")
        return

    try:
        summary = summarizer(context, max_length=150, min_length=30, do_sample=False)
        answer = summary[0]["summary_text"]
    except Exception:
        answer = "Answer not found in the book."

    st.subheader("Answer")
    st.write(f"{answer}\n\n📄 (Derived from page(s): {', '.join(map(str, pages_used))})")

    st.subheader("Relevant Book Chunks with Page Numbers")
    for i, d in enumerate(docs):
        st.write(f"**Chunk {i+1} (Page {d.metadata.get('page', 'N/A')}):** {d.page_content[:500]}...")


def main():
    st.set_page_config(page_title="Book QA System")
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

if __name__ == "__main__":
    main()
