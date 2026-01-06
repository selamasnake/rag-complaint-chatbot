# Notebooks Overview

This folder contains Jupyter notebooks for the CrediTrust RAG-powered complaint analysis project.

## Notebooks
- `eda_preprocessing.ipynb`: Exploratory data analysis and preprocessing of CFPB complaint data.  
  Steps include:
  - Load raw complaints
  - Analyze product distributions and narrative lengths
  - Identify missing narratives
  - Filter relevant product categories
  - Clean text narratives
  - Save processed dataset for downstream embedding and RAG tasks
  
- `chunking_embedding.ipynb`:  
  Transform cleaned complaints into embeddings and build a FAISS vector store.  
  Steps include:
  - Stratified sampling of complaints across product categories
  - Text chunking to improve embedding quality
  - Generate vector embeddings using sentence-transformers
  - Build and save FAISS index with metadata for semantic search