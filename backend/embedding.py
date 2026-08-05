"""
backend/embedding.py
--------------------------------------------------

Generates embeddings for PDF chunks using
Sentence Transformers.

Pipeline:

PDFs
   ↓
PDFLoader
   ↓
TextSplitter
   ↓
Chunks
   ↓
SentenceTransformer
   ↓
Embeddings

Author: ACB AI Assistant
"""

from sentence_transformers import SentenceTransformer
from pdf_loader import PDFLoader
from text_splitter import TextSplitter


class EmbeddingGenerator:
    """
    Generates embeddings for document chunks.
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-base-en-v1.5",
    ):
        """
        Initialize the embedding model.
        """

        print("=" * 60)
        print("Loading Embedding Model...")
        print("=" * 60)

        self.model = SentenceTransformer(model_name)

        print(f"Model Loaded: {model_name}")
        print("=" * 60)

    def generate_embeddings(
        self,
        chunks: list[dict],
    ):
        """
        Generate embeddings for every chunk.

        Args:
            chunks (list):
                List of chunk dictionaries.

        Returns:
            tuple:
                (
                    embeddings,
                    chunks
                )
        """

        texts = [
            f"""
        Document: {chunk['filename']}
        Page: {chunk['page']}

        Content:
        {chunk['text']}
        """.strip()
        for chunk in chunks
        ]

        print()
        print("=" * 60)
        print("Generating Embeddings...")
        print("=" * 60)

        embeddings = self.model.encode(
            texts,
            batch_size=32,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        print()
        print("=" * 60)
        print("Embedding Generation Completed")
        print("=" * 60)
        print(f"Chunks Processed : {len(chunks)}")
        print(f"Embedding Shape  : {embeddings.shape}")
        print("=" * 60)

        return embeddings, chunks


if __name__ == "__main__":

    # -------------------------------------------------
    # Load PDFs
    # -------------------------------------------------

    loader = PDFLoader("data/pdfs")
    documents = loader.load_documents()

    # -------------------------------------------------
    # Split into chunks
    # -------------------------------------------------

    splitter = TextSplitter()

    chunks = splitter.split_documents(documents)

    # -------------------------------------------------
    # Generate embeddings
    # -------------------------------------------------

    generator = EmbeddingGenerator()

    embeddings, chunks = generator.generate_embeddings(chunks)

    # -------------------------------------------------
    # Preview
    # -------------------------------------------------

    print("\nFirst Chunk")
    print("-" * 60)

    print(f"ID      : {chunks[0]['chunk_id']}")
    print(f"Source  : {chunks[0]['source']}")

    print("\nText:")
    print(chunks[0]["text"][:250])

    print("\nEmbedding Dimension:")
    print(len(embeddings[0]))

    print("\nFirst 10 Values:")

    print(embeddings[0][:10])

    print("\n" + "=" * 60)
    print("Embedding test completed successfully!")
    print("=" * 60)