"""
backend/llm.py
--------------------------------------------------

Google Gemini LLM Wrapper

Uses Gemini 2.5 Flash to generate answers
from retrieved RAG context.

Author: ACB AI Assistant
"""

import os

from dotenv import load_dotenv
from pathlib import Path
from google import genai


class GeminiLLM:
    """
    Gemini wrapper for RAG.
    """

    def __init__(
        self,
        model_name: str = "gemini-3.5-flash",
    ):

        print("=" * 60)
        print("Loading Gemini...")
        print("=" * 60)

        # Project root = one folder above "backend"
        PROJECT_ROOT = Path(__file__).resolve().parent.parent

        # Load .env explicitly
        load_dotenv(PROJECT_ROOT / ".env")

        print("Loading .env from:", PROJECT_ROOT / ".env")
        print("Exists:", (PROJECT_ROOT / ".env").exists())

        api_key = os.getenv("GOOGLE_API_KEY")

        if not api_key:
            raise ValueError(
                "GOOGLE_API_KEY not found in .env"
            )

        self.client = genai.Client(api_key=api_key)

        self.model_name = model_name

        print(f"Model Loaded : {self.model_name}")
        print("=" * 60)

    # ----------------------------------------------------

    def build_prompt(
        self,
        question: str,
        documents: list,
    ) -> str:
        """
        Build RAG prompt.
        """

        context = ""

        for i, doc in enumerate(documents, start=1):

            context += (
                f"\n========== DOCUMENT {i} ==========\n"
                f"Source : {doc['source']}\n\n"
                f"{doc['text']}\n"
            )

        prompt = f"""
You are Schneider Electric's MasterPacT MTZ AI Assistant.

You answer ONLY from the documentation below.

Rules:

1. Never invent information.

2. If the answer is not contained in the documents,
reply:

"I couldn't find this information in the provided documentation."

3. Keep technical terminology exactly as written.

4. Answer in Markdown.

5. End your answer with:

Sources:
- source names

====================================================

DOCUMENTATION

{context}

====================================================

QUESTION

{question}

====================================================

ANSWER
"""

        return prompt

    # ----------------------------------------------------

    def generate_answer(
        self,
        question: str,
        documents: list,
    ) -> str:
        """
        Generate answer from Gemini.
        """

        prompt = self.build_prompt(
            question,
            documents,
        )

        response = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt,
        )

        return response.text


# ======================================================
# Test
# ======================================================
if __name__ == "__main__":

    llm = GeminiLLM()

    docs = [
        {
            "source": "MTZ Active Catalogue - Page 31",
            "text": """
A wide range of accessories can be used
to improve the functions of control
and monitoring.
"""
        }
    ]

    question = "What accessories are available?"

    print("\nGenerating answer...\n")

    answer = llm.generate_answer(
        question=question,
        documents=docs,
    )

    print("=" * 60)
    print("Gemini Response")
    print("=" * 60)
    print(answer)