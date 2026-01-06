# rag-complaint-chatbot

This project develops a Retrieval-Augmented Generation (RAG) chatbot to analyze customer complaints for CrediTrust Financial.  
It transforms raw, unstructured complaint data into actionable insights for product, support, and compliance teams, enabling them to identify trends and emerging issues quickly.  

#### Project Structure
- `data/` — contains raw CFPB complaint data and processed datasets (ignored in git).  

- `notebooks/` — Jupyter notebooks for analysis and preprocessing:

    - `eda_preprocessing.ipynb` — exploratory data analysis, complaint distributions, narrative length analysis, and text cleaning.  

- `src/` — Python modules for core preprocessing and EDA logic:

    - `data_preprocessing.py` — classes to load, filter, clean, and perform EDA on complaint datasets.  

- `.github/workflows/unittests.yml` — CI workflow for automated testing.  

#### How to Use
1. Clone the repository and install dependencies from `requirements.txt`.  
2. Use the notebooks to explore and preprocess complaint data.  
3. The cleaned dataset in `data/processed/` is ready for embedding and vector store creation.  
4. The RAG chatbot can then query the processed data to answer questions based on real customer complaints.  

#### Requirements
See `requirements.txt` for libraries including `pandas`, `numpy`, `matplotlib`, `seaborn` used for preprocessing and EDA.
