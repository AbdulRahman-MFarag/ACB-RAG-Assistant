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
import re

class TextSplitter:
    """
    Splits extracted PDF pages into overlapping chunks.
    """

    def __init__(
        self,
        chunk_size: int = 600,
        chunk_overlap: int = 120,
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
              ";",
              ":",
              ",",
              " ",
              ""
            ]
        )

    def clean_text(self, text: str) -> str:
        """
        Clean extracted PDF text before chunking.
        """

        # Normalize line endings
        text = text.replace("\r", "\n")

        # Remove repeated spaces
        text = re.sub(r"[ \t]+", " ", text)

        # Remove multiple blank lines
        text = re.sub(r"\n{3,}", "\n\n", text)

        lines = []

        for line in text.split("\n"):

            line = line.strip()

            if not line:
                continue

            # Remove page numbers
            if re.fullmatch(r"\d+", line):
                continue

            # Remove isolated letters
            if re.fullmatch(r"[A-Z]", line):
                continue

            # Remove figure labels
            if re.fullmatch(r"[A-Z]\d+", line):
                continue

            if re.fullmatch(r"Figure\s+\d+\.?", line):
                continue

            if re.fullmatch(r"Fig\.\s*\d+", line):
                continue

            if re.fullmatch(r"[A-Za-z]", line):
                continue
            # Remove copyright/footer lines
            if line == "Schneider Electric":
                continue

            if "All rights reserved" in line:
                continue

            if line == "www.se.com":
                continue

            if "DOCA0" in line:
                continue

            # Remove page references like "Page 23"
            if re.fullmatch(r"Page\s+\d+", line):
                continue

            lines.append(line)

        text = "\n".join(lines)

        return text.strip()

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

            cleaned_text = self.clean_text(document["text"])

            if not cleaned_text:
                continue
            split_text = self.splitter.split_text(cleaned_text)

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
                        "length": len(piece.strip()),
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
        chunk_size=600,
        chunk_overlap=120,
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