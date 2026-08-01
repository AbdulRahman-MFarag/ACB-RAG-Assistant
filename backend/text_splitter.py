"""
backend/text_splitter.py
--------------------------------------------------

Splits extracted PDF pages into smaller overlapping
chunks for semantic search (RAG).

Input:
[
    {
        "filename": "...",
        "filepath": "...",
        "page": 1,
        "text": "..."
    }
]

Output:
[
    {
        "chunk_id": "...",
        "filename": "...",
        "filepath": "...",
        "page": 1,
        "source": "...",
        "text": "..."
    }
]

Author: ACB AI Assistant
"""

from langchain_text_splitters import RecursiveCharacterTextSplitter
from pdf_loader import PDFLoader


class TextSplitter:
    """
    Splits extracted PDF pages into overlapping chunks.
    """

    def __init__(
        self,
        chunk_size: int = 800,
        chunk_overlap: int = 150,
    ):
        """
        Initialize the text splitter.

        Args:
            chunk_size (int): Maximum number of characters per chunk.
            chunk_overlap (int): Number of overlapping characters.
        """

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=[
                "\n\n",
                "\n",
                ". ",
                " ",
                ""
            ]
        )

    def split_documents(self, documents: list[dict]) -> list[dict]:
        """
        Split PDF pages into overlapping chunks.

        Args:
            documents (list): Output from PDFLoader.

        Returns:
            list: Chunked documents with metadata.
        """

        chunks = []

        print("=" * 60)
        print("Splitting documents into chunks...")
        print("=" * 60)

        for document in documents:

            split_text = self.splitter.split_text(document["text"])

            for index, piece in enumerate(split_text, start=1):

                chunk_id = (
                    f"{document['filename']}"
                    f"_p{document['page']}"
                    f"_c{index}"
                )

                chunks.append(
                    {
                        "chunk_id": chunk_id,
                        "filename": document["filename"],
                        "filepath": document["filepath"],
                        "page": document["page"],
                        "source": f"{document['filename']} - Page {document['page']}",
                        "text": piece.strip(),
                    }
                )

        print()
        print("=" * 60)
        print("Chunking Completed")
        print("=" * 60)
        print(f"Original Pages   : {len(documents)}")
        print(f"Generated Chunks : {len(chunks)}")
        print("=" * 60)

        return chunks


if __name__ == "__main__":

    # ---------------------------------------------------
    # Load PDF Pages
    # ---------------------------------------------------

    loader = PDFLoader("data/pdfs")
    documents = loader.load_documents()

    # ---------------------------------------------------
    # Split into Chunks
    # ---------------------------------------------------

    splitter = TextSplitter(
        chunk_size=800,
        chunk_overlap=150,
    )

    chunks = splitter.split_documents(documents)

    # ---------------------------------------------------
    # Preview First Chunk
    # ---------------------------------------------------

    first = chunks[0]

    print("\nFirst Chunk")
    print("-" * 60)

    print(f"Chunk ID : {first['chunk_id']}")
    print(f"File     : {first['filename']}")
    print(f"Page     : {first['page']}")
    print(f"Source   : {first['source']}")

    print("\nChunk Text:\n")
    print(first["text"])

    # ---------------------------------------------------
    # Preview Last Chunk
    # ---------------------------------------------------

    last = chunks[-1]

    print("\n" + "=" * 60)

    print("Last Chunk")
    print("-" * 60)

    print(f"Chunk ID : {last['chunk_id']}")
    print(f"File     : {last['filename']}")
    print(f"Page     : {last['page']}")
    print(f"Source   : {last['source']}")

    print("\nChunk Text:\n")
    print(last["text"])

    print("\n" + "=" * 60)
    print("Chunking test completed successfully!")
    print("=" * 60)