"""
backend/pdf_loader.py
--------------------------------------------------

Reads all PDF files from the data/pdfs folder and
extracts text page by page.

Each page is stored as:

{
    "filename": "...",
    "filepath": "...",
    "page": 1,
    "text": "..."
}

This file is responsible ONLY for reading PDFs.
Chunking, embeddings, and vector storage are handled
in later modules.

Author: ACB AI Assistant
"""

import os
import fitz  # PyMuPDF


class PDFLoader:
    """
    Loads PDF documents and extracts text page by page.
    """

    def __init__(self, pdf_folder: str):
        """
        Initialize the PDF loader.

        Args:
            pdf_folder (str): Path to the folder containing PDF files.
        """
        self.pdf_folder = pdf_folder

    def load_documents(self) -> list[dict]:
        """
        Read every PDF in the folder.

        Returns:
            list[dict]: A list of dictionaries containing:
                - filename
                - filepath
                - page
                - text
        """

        documents = []

        if not os.path.exists(self.pdf_folder):
            raise FileNotFoundError(
                f"Folder not found: {self.pdf_folder}"
            )

        pdf_files = sorted(
            [
                file
                for file in os.listdir(self.pdf_folder)
                if file.lower().endswith(".pdf")
            ]
        )

        if not pdf_files:
            raise FileNotFoundError(
                "No PDF files were found inside the folder."
            )

        print("=" * 60)
        print("Loading PDF documents...")
        print("=" * 60)

        total_pages = 0

        for file in pdf_files:

            pdf_path = os.path.join(self.pdf_folder, file)

            print(f"[INFO] Reading: {file}")

            try:
                pdf = fitz.open(pdf_path)

                for page_number, page in enumerate(pdf, start=1):

                    text = page.get_text().strip()

                    if not text:
                        continue

                    documents.append(
                        {
                            "filename": file,
                            "filepath": pdf_path,
                            "page": page_number,
                            "text": text,
                        }
                    )

                    total_pages += 1

                pdf.close()

            except Exception as e:
                print(f"[ERROR] Failed to read '{file}'")
                print(e)

        print()
        print("=" * 60)
        print("PDF Loading Completed")
        print("=" * 60)
        print(f"PDF Files Loaded : {len(pdf_files)}")
        print(f"Pages Extracted  : {total_pages}")
        print(f"Documents Stored : {len(documents)}")
        print("=" * 60)

        return documents


if __name__ == "__main__":

    loader = PDFLoader("data/pdfs")

    documents = loader.load_documents()

    if documents:
        print("\nFirst Extracted Document")
        print("-" * 60)

        print(f"File : {documents[0]['filename']}")
        print(f"Page : {documents[0]['page']}")

        print("\nPreview:\n")
        print(documents[0]["text"][:1000])

    else:
        print("No text was extracted from the PDFs.")