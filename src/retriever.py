import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
import pandas as pd


class FaissComplaintRetriever:
    """
    Handles semantic retrieval over complaint chunks using FAISS.
    Ensures metadata aligns with vectors even for chunked text.
    """

    def __init__(
        self,
        index_path: str,
        metadata_path: str,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    ):
        # Load embedder
        self.embedder = SentenceTransformer(embedding_model)

        # Load FAISS index
        self.index = faiss.read_index(index_path)

        # Load metadata
        with open(metadata_path, "rb") as f:
            meta = pickle.load(f)
            # Convert list of dicts to DataFrame if needed
            if isinstance(meta, list):
                self.metadata = pd.DataFrame(meta)
            elif isinstance(meta, pd.DataFrame):
                self.metadata = meta
            else:
                raise ValueError("Unsupported metadata format")

        # Safety check: clip FAISS index to metadata length if needed
        if self.index.ntotal > len(self.metadata):
            print(
                f"Warning: FAISS index has {self.index.ntotal} vectors, "
                f"but metadata has {len(self.metadata)} records."
                " Clipping search to metadata size."
            )
            self.max_index = len(self.metadata)
        else:
            self.max_index = self.index.ntotal

    def retrieve(self, query: str, k: int = 5):
        """
        Returns top-k relevant complaint chunks and metadata.
        """
        # 1️⃣ Embed the query
        query_embedding = self.embedder.encode([query]).astype("float32")

        # 2️⃣ FAISS search
        distances, indices = self.index.search(query_embedding, k)

        documents = []
        metadatas = []

        # 3️⃣ Safely fetch metadata for retrieved vectors
        for idx in indices[0]:
            if idx >= self.max_index:
                # Skip vectors without metadata
                continue
            record = self.metadata.iloc[idx].to_dict()
            documents.append(record["chunk_text"]) 
            metadatas.append(record)

        return documents, metadatas
