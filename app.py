import streamlit as st
import json
import pdfplumber
import re
from transformers import pipeline
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

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

@st.cache_resource
def load_sentence_transformer():
    return SentenceTransformer('all-MiniLM-L6-v2')

summarizer = load_summarizer()
sentence_model = load_sentence_transformer()

def is_question_relevant(question, context):
    """
    Balanced relevance checking with adjustable thresholds
    """
    # If context is too short, it's likely not relevant
    if len(context) < 30:
        return False
    
    # Calculate semantic similarity
    question_embedding = sentence_model.encode([question])
    context_embedding = sentence_model.encode([context])
    similarity_score = cosine_similarity(question_embedding, context_embedding)[0][0]
    
    # Check for keyword matches
    question_lower = question.lower()
    context_lower = context.lower()
    
    # Extract meaningful keywords (excluding common words)
    stop_words = {'what', 'is', 'the', 'a', 'an', 'how', 'why', 'when', 'where', 'who', 'does', 'do'}
    question_keywords = [word for word in question_lower.split() if word not in stop_words and len(word) > 2]
    
    # If no specific keywords, be more lenient
    if not question_keywords:
        return similarity_score > 0.3
    
    # Count how many question keywords appear in context
    matching_keywords = sum(1 for keyword in question_keywords if keyword in context_lower)
    keyword_ratio = matching_keywords / len(question_keywords)
    
    # Combined relevance score (weighted average)
    relevance_score = 0.6 * similarity_score + 0.4 * keyword_ratio
    
    # Adjust threshold based on question type
    threshold = 0.35  # Lowered threshold to be more inclusive
    
    # For definition questions (what is X), be more lenient
    if question_lower.startswith(('what is', 'what are', 'define')):
        threshold = 0.25
    
    return relevance_score > threshold

def user_input(user_question):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    try:
        db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    except Exception:
        st.error("FAISS index not found. Please process a book first.")
        return

    # First try a direct search
    docs = db.similarity_search(user_question, k=3)
    
    # If no good results, try with a broader search
    if not docs or all(len(d.page_content) < 50 for d in docs):
        docs = db.similarity_search(user_question, k=5)
    
    if not docs:
        st.info("No relevant content found in the book for this question.")
        return

    # Create context from retrieved docs
    context = "\n".join([d.page_content for d in docs])
    
    # Check if the retrieved content is actually relevant to the question
    if not is_question_relevant(user_question, context):
        # Give one more chance with a different approach
        alternative_docs = db.similarity_search(" ".join(user_question.split()[-2:]), k=3)  # Search on last two words
        alternative_context = "\n".join([d.page_content for d in alternative_docs])
        
        if not is_question_relevant(user_question, alternative_context):
            st.info("I'm sorry, but this question doesn't seem to be covered in the book content. Please ask a question related to the PDF content.")
            return
        else:
            # Use the alternative context if it's relevant
            context = alternative_context
            docs = alternative_docs

    try:
        summary = summarizer(context, max_length=150, min_length=30, do_sample=False)
        answer = summary[0]["summary_text"]
        
        # Final verification - but be more lenient here
        if not is_question_relevant(user_question, answer):
            # Even if the answer doesn't perfectly match, show it if the context was relevant
            st.warning("The book may not have a direct answer to your question, but here's some related information:")
            
    except Exception:
        answer = "Answer not found in the book."

    # Collect pages for final answer
    pages_used = sorted(set([d.metadata.get("page", "N/A") for d in docs]))

    st.subheader("Answer")
    st.write(f"{answer}\n\n📄 (Derived from page(s): {', '.join(map(str, pages_used))})")




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
