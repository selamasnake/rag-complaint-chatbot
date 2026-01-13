from retriever import FaissComplaintRetriever
from generator import AnswerGenerator


class RAGPipeline:
    """
    End-to-end RAG pipeline: retrieve → generate → return sources.
    """

    def __init__(
        self,
        index_path="vector_store/index.faiss",
        metadata_path="vector_store/metadata.pkl"
    ):
        self.retriever = FaissComplaintRetriever(
            index_path=index_path,
            metadata_path=metadata_path
        )
        self.generator = AnswerGenerator()

    def run(self, question: str, k: int = 5):
        documents, metadata = self.retriever.retrieve(question, k)

        answer = self.generator.generate(question, documents)

        sources = [
            {
                "text": doc,
                "product_category": meta.get("product_category"),
                "complaint_id": meta.get("complaint_id"),
                "chunk_index": meta.get("chunk_index"),
                "total_chunks": meta.get("total_chunks")
            }
            for doc, meta in zip(documents, metadata)
        ]

        return answer, sources
