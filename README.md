# rag-complaint-chatbot

This project develops a Retrieval-Augmented Generation (RAG) chatbot to analyze customer complaints for CrediTrust Financial.  
It transforms raw, unstructured complaint data into actionable insights for product, support, and compliance teams, enabling them to identify trends and emerging issues quickly.  

#### Project Structure
- `data/` — contains raw CFPB complaint data and processed datasets (ignored in git).  

- `notebooks/` — Jupyter notebooks for analysis and preprocessing:

    - `eda_preprocessing.ipynb` — exploratory data analysis, complaint distributions, narrative length analysis, and text cleaning.  
    - `chunking_embedding.ipynb` — stratified sampling, chunking, embedding, and vector store construction.
    - `rag_pipeline.ipynb` — End-to-end execution and evaluation of the RAG pipeline.

- `src/` — Python modules for core preprocessing and EDA logic:

    - `data_preprocessing.py` — classes to load, filter, clean, and perform EDA on complaint datasets.  
    - `chunking_embedding.py` — sampling, chunking, embedding, and vector store functionality.
    - `retriever.py` — Retrieves semantically relevant complaint text and metadata using FAISS.
    - `generator.py` — Generates grounded analytical answers from retrieved complaint excerpts.
    - `rag_pipeline.py` — Orchestrates retrieval, generation, and source attribution.
    - `prompts.py` — Defines the structured prompt template for retrieval-augmented generation.

- `.github/workflows/unittests.yml` — CI workflow for automated testing.  

#### How to Use
1. Clone the repository and install dependencies from `requirements.txt`.  
2. Use the notebooks to explore and preprocess complaint data.  
3. The cleaned dataset in `data/processed/` is ready for embedding and vector store creation.  
4. The RAG chatbot can then query the processed data to answer questions based on real customer complaints.  

#### Requirements
See `requirements.txt` for libraries including `pandas`, `numpy`, `matplotlib`, `seaborn`,  `sentence-transformers` `faiss-cpu`, `langchain`.
