"""
backend/chatbot.py
--------------------------------------------------

Complete RAG Chatbot

User Question
      ↓
Retriever
      ↓
Top Documents
      ↓
Gemini
      ↓
Final Answer
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

        docs = self.retriever.search(
            question,
            top_k=5,
        )

        answer = self.llm.generate_answer(
            question,
            docs,
        )

        return answer


if __name__ == "__main__":

    chatbot = ACBChatbot()

    while True:

        print()

        question = input("Ask a question (or type exit): ")

        if question.lower() == "exit":
            break

        print("\nSearching documentation...\n")

        answer = chatbot.ask(question)

        print("=" * 60)
        print(answer)
        print("=" * 60)