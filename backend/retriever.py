"""
backend/retriever.py
--------------------------------------------------

Semantic search using FAISS.

Author: ACB AI Assistant
"""

import numpy as np

from sentence_transformers import SentenceTransformer

from vector_store import VectorStore


class Retriever:
    """
    Retrieves the most relevant chunks
    from the FAISS vector store.
    """

    def __init__(
    self,
    model_name="BAAI/bge-base-en-v1.5",
):

        print("=" * 60)
        print("Loading Retriever...")
        print("=" * 60)

        self.model = SentenceTransformer(model_name)

        store = VectorStore()

        self.index, self.metadata = store.load_index()

        print(f"Loaded {len(self.metadata)} chunks.")

        print("=" * 60)

    def search(
        self,
        query: str,
        top_k: int = 5,
    ):

        query_embedding = self.model.encode(
    [
        f"Represent this sentence for searching relevant passages: {query}"
    ],
    convert_to_numpy=True,
    normalize_embeddings=True,
)
        
        print("Query dimension:", query_embedding.shape)

        distances, indices = self.index.search(
            query_embedding.astype("float32"),
            top_k,
        )

        results = []

        for score, idx in zip(
            distances[0],
            indices[0],
        ):

            document = self.metadata[idx].copy()

            document["score"] = float(score)

            results.append(document)

        return results


if __name__ == "__main__":

    retriever = Retriever()

    while True:

        print()

        question = input("Ask a question (or type exit): ")

        if question.lower() == "exit":
            break

        results = retriever.search(question)

        print("\nTop Results")

        print("=" * 60)

        for i, result in enumerate(results, start=1):

            print(f"\nResult {i}")
            print("-" * 60)

            print(f"Score : {result['score']:.4f}")

            print(f"Source: {result['source']}")

            print()

            print(result["text"][:400])