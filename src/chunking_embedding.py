import os
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sklearn.model_selection import train_test_split

class ComplaintSampler:
    """Stratified sampling across product categories."""
    def __init__(self, df: pd.DataFrame, product_col: str = "product_category"):
        self.df = df
        self.product_col = product_col

    def stratified_sample(self, n_samples: int = 15000, random_state: int = 42) -> pd.DataFrame:
        """Returns a stratified sample proportional to product counts using train_test_split."""
        # Using sklearn's train_test_split is the robust way to ensure exact stratified counts
        df_sampled, _ = train_test_split(
            self.df, 
            train_size=n_samples, 
            stratify=self.df[self.product_col], 
            random_state=random_state
        )
        return df_sampled.reset_index(drop=True)

class TextChunker:
    """Splits text into semantic chunks using RecursiveCharacterTextSplitter."""
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        # Recursive splitting prioritizes paragraph and sentence boundaries
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""]
        )

    def chunk_dataframe(self, df: pd.DataFrame, text_col: str = "cleaned_narrative") -> pd.DataFrame:
        """Returns a DataFrame with chunked text and metadata for tracing."""
        all_chunks = []
        for _, row in df.iterrows():
            text = str(row[text_col])
            chunks = self.splitter.split_text(text)
            for idx, chunk in enumerate(chunks):
                all_chunks.append({
                    "complaint_id": row.get("Complaint ID"),
                    "product_category": row.get("Product"),
                    "chunk_text": chunk,
                    "chunk_index": idx,
                    "total_chunks": len(chunks)
                })
        return pd.DataFrame(all_chunks)

class VectorStoreBuilder:
    """Build embeddings and FAISS vector store with Cosine Similarity."""
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def create_embeddings(self, df: pd.DataFrame, text_col: str = "chunk_text") -> np.ndarray:
        """Generate normalized embeddings for all text chunks."""
        embeddings = self.model.encode(
            df[text_col].tolist(),
            show_progress_bar=True,
            normalize_embeddings=True # Normalizing is key for Cosine Similarity
        )
        return embeddings

    def build_faiss_index(self, embeddings: np.ndarray, df: pd.DataFrame, save_path: str = "../vector_store") -> faiss.Index:
        """Build FAISS IndexFlatIP and save metadata."""
        dim = embeddings.shape[1]
        # IndexFlatIP + Normalized Embeddings = Cosine Similarity
        index = faiss.IndexFlatIP(dim)
        index.add(embeddings.astype('float32'))

        os.makedirs(save_path, exist_ok=True)
        faiss.write_index(index, os.path.join(save_path, "index.faiss"))
        # Save metadata as parquet for better performance with 1.3M+ records
        df.to_parquet(os.path.join(save_path, "metadata.parquet"), index=False)
        return index

class VectorStoreSearcher:
    """Handles loading and querying the FAISS index with metadata."""
    
    def __init__(self, index_path: str = "../vector_store", model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        # Load the index and metadata
        self.index = faiss.read_index(os.path.join(index_path, "index.faiss"))
        self.metadata = pd.read_parquet(os.path.join(index_path, "metadata.parquet"))

    def search(self, query: str, k: int = 3) -> pd.DataFrame:
        """Performs semantic search and returns a DataFrame of results."""
        # Encode and normalize query
        query_vec = self.model.encode([query], normalize_embeddings=True)
        
        # Search index
        distances, indices = self.index.search(query_vec.astype('float32'), k)
        
        # Retrieve corresponding metadata rows
        results = self.metadata.iloc[indices[0]].copy()
        results['score'] = distances[0]
        return results