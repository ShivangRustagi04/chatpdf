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

OR 

run live:-

live link - http://0198bdb9-65b2-e341-0067-5746e637d873.share.connect.posit.cloud:80


```




## Sample Input and Output:

1. what is ecosystem?
answer = An ecosystem can be visualised as a functional unit of 12.2.Productivity nature, where living organisms interact among themselves and also with the surrounding physical environment. Ecosystem varies greatly in size from a small pond to a large forest or a sea. Many ecologists regard the entire 12.4 Energy Flow biosphere as a global ecosystem, as a composite of all local ecosystems on Earth.

2. what is primary production?
answer = Primary productivity depends on the plant species inhabiting a particular area. It also depends on a variety of environmental factors, availability of nutrients and photosynthetic capacity of plants. Productivity, decomposition, energy flow, and nutrient cycling are the four important components of an ecosystem.

3. what is productivity?
answer = The rate of biomass production is called productivity. It is expressed in terms of gm2 yr1 or (kcal m2) yr1 to compare the productivity of different ecosystems. Gross primary productivity of an ecosystem is the rate of production of organic matter during photosynthesis. Primary productivity depends on the plant species inhabiting a particular area. It also depends on a variety of environmental factors.

4. what is decomposition?
answer = Dead plant remains such as leaves, bark, flowers and dead remains of animals, including fecal matter, constitute detritus, which is the raw material for decomposition. The important steps in the process of decomposition are fragmentation, leaching, catabolism, humification and mineralisation.

5. what are producers?
answer = The green plant in the ecosystem are called producers. In a terrestrial ecosystem, major producers are herbaceous and woody plants. Producers in an aquatic ecosystem are various species like phytoplankton, algae and higher plants. Starting from the plants (or producers) food chains or rather webs are formed such that an animal feeds on a plant or on another animal and in turn is food for another.
## Setup & Installation Instructions
1. Open the streamlit app or directly go to live link mentioned in Setup and & Installation
2. Upload PDFs and Process
3. Ask Questions

## ⚠️ Limitations & Security Compliance

File Size: Large PDF files (>50MB) may slow down processing or exceed memory limits.

Text Extraction: PDFs with scanned images or complex formatting may result in incomplete text extraction.

Summarization Limits: The summarizer may truncate or generalize context; verify with original content if precision is critical.

Security: Uploaded PDFs are stored only temporarily for processing. Ensure sensitive documents are handled carefully.

Offline Use: The app is designed for local/private deployment; no cloud storage or external API calls are made by default.