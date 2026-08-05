"""
backend/chatbot.py
--------------------------------------------------

Complete RAG Chatbot

Author: ACB AI Assistant
"""

from retriever import Retriever
from llm import GeminiLLM


class ACBChatbot:

    def __init__(self):

        print("=" * 60)
        print("Starting ACB AI Assistant...")
        print("=" * 60)

        self.retriever = Retriever()
        self.llm = GeminiLLM()

        print("\nSystem Ready!")
        print("=" * 60)

    def ask(self, question: str):

        # Retrieve relevant documents
        documents = self.retriever.search(
            question,
            top_k=5,
        )

        # No documents found
        if len(documents) == 0:

            return {
                "answer": "I couldn't find any relevant information in the documentation.",
                "sources": [],
                "documents": [],
            }

        # Generate answer
        answer = self.llm.generate_answer(
            question,
            documents,
        )

        # Extract unique sources
        sources = []

        seen = set()

        for doc in documents:

            if doc["source"] not in seen:

                seen.add(doc["source"])

                sources.append(doc["source"])

        return {

            "answer": answer,

            "sources": sources,

            "documents": documents,
        }


# ==========================================================
# Test
# ==========================================================

if __name__ == "__main__":

    chatbot = ACBChatbot()

    while True:

        print()

        question = input("Ask a question (or type exit): ")

        if question.lower() == "exit":
            break

        response = chatbot.ask(question)

        print()

        print("=" * 60)

        print(response["answer"])

        print()

        print("Sources:")

        for source in response["sources"]:

            print("-", source)

        print("=" * 60)