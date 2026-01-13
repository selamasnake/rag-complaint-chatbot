# Source Modules Overview

This folder contains Python modules for data preprocessing, EDA, and related utilities.

## Modules

- `data_preprocessing.py`:  
  Load, clean, and perform exploratory analysis on CFPB complaint datasets.
- `chunking_embedding.py`:  
  Stratified sampling, text chunking, embedding generation, and FAISS vector store management for complaint narratives.
- `retriever.py` — Retrieves semantically relevant complaint text and aligned metadata using FAISS.

- `generator.py` — Generates grounded analytical answers from retrieved complaint excerpts using an LLM.

- `rag_pipeline.py` — Orchestrates retrieval, generation, and source attribution in an end-to-end RAG workflow.

- `prompts.py` — Defines the structured prompt template used to guide retrieval-augmented generation.