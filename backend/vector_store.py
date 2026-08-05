"""
backend/vector_store.py
--------------------------------------------------

Creates, saves, and loads a FAISS vector index.

Author: ACB AI Assistant
"""

import os
import pickle

import faiss
import numpy as np

from pdf_loader import PDFLoader
from text_splitter import TextSplitter
from embedding import EmbeddingGenerator


class VectorStore:
    """
    Handles FAISS vector index operations.
    """

    def __init__(
        self,
        index_path: str = "data/vector_store/faiss.index",
        metadata_path: str = "data/vector_store/metadata.pkl",
    ):

        self.index_path = index_path
        self.metadata_path = metadata_path

        os.makedirs(os.path.dirname(index_path), exist_ok=True)

    def build_index(
        self,
        embeddings: np.ndarray,
    ) -> faiss.Index:

        print("=" * 60)
        print("Building FAISS Index...")
        print("=" * 60)

        dimension = embeddings.shape[1]

        index = faiss.IndexFlatIP(dimension)

        print("Embedding dimension:", embeddings.shape[1])
        index.add(embeddings.astype("float32"))

        print(f"Vectors Indexed : {index.ntotal}")

        print("=" * 60)

        return index

    def save_index(
        self,
        index: faiss.Index,
        metadata: list,
    ):

        faiss.write_index(index, self.index_path)

        with open(self.metadata_path, "wb") as file:
            pickle.dump(metadata, file)

        print("Vector store saved successfully.")

        print(f"Index    : {self.index_path}")
        print(f"Metadata : {self.metadata_path}")

    def load_index(self):

        if not os.path.exists(self.index_path):
            raise FileNotFoundError("FAISS index not found.")

        if not os.path.exists(self.metadata_path):
            raise FileNotFoundError("Metadata file not found.")

        index = faiss.read_index(self.index_path)

        with open(self.metadata_path, "rb") as file:
            metadata = pickle.load(file)

        print("=" * 60)
        print("Vector Store Loaded")
        print("=" * 60)

        print(f"Vectors : {index.ntotal}")

        return index, metadata


if __name__ == "__main__":

    # ----------------------------------------
    # Load PDFs
    # ----------------------------------------

    loader = PDFLoader("data/pdfs")

    documents = loader.load_documents()

    # ----------------------------------------
    # Split Documents
    # ----------------------------------------

    splitter = TextSplitter()

    chunks = splitter.split_documents(documents)

    # ----------------------------------------
    # Generate Embeddings
    # ----------------------------------------

    generator = EmbeddingGenerator()

    embeddings, metadata = generator.generate_embeddings(chunks)

    # ----------------------------------------
    # Build FAISS
    # ----------------------------------------

    store = VectorStore()

    index = store.build_index(embeddings)

    store.save_index(index, metadata)

    # ----------------------------------------
    # Reload Test
    # ----------------------------------------

    index, metadata = store.load_index()

    print()

    print("=" * 60)

    print("Reload Test")

    print("=" * 60)

    print(f"Vectors : {index.ntotal}")

    print(f"Metadata: {len(metadata)}")

    print()

    print("First Document")

    print("-" * 60)

    print(metadata[0]["source"])

    print()

    print(metadata[0]["text"][:250])

    print()

    print("=" * 60)

    print("FAISS test completed successfully!")

    print("=" * 60)