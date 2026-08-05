"""
frontend/app.py
--------------------------------------------------

Main Streamlit GUI for the ACB AI Assistant

Author: ACB AI Assistant
"""

import sys
from pathlib import Path

import streamlit as st

# -------------------------------------------------------
# Allow importing backend modules
# -------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT / "backend"))

from chatbot import ACBChatbot

# -------------------------------------------------------
# Page Configuration
# -------------------------------------------------------

st.set_page_config(
    page_title="ACB AI Assistant",
    page_icon="⚡",
    layout="wide",
)

# -------------------------------------------------------
# Load CSS
# -------------------------------------------------------

css_file = Path(__file__).parent / "style.css"

if css_file.exists():
    with open(css_file) as f:
        st.markdown(
            f"<style>{f.read()}</style>",
            unsafe_allow_html=True,
        )

# -------------------------------------------------------
# Session State
# -------------------------------------------------------

if "chatbot" not in st.session_state:
    with st.spinner("Loading AI Assistant..."):
        st.session_state.chatbot = ACBChatbot()

if "messages" not in st.session_state:
    st.session_state.messages = []

# -------------------------------------------------------
# Sidebar
# -------------------------------------------------------

with st.sidebar:

    st.title("⚡ ACB AI Assistant")

    st.markdown("---")

    st.markdown(
        """
This AI Assistant is designed to answer questions about Schneider Electric MasterPacT MTZ documentation. 
It uses a RAG approach to provide accurate and relevant answers based on the available documentation.
        """
    )

    st.markdown("---")

    if st.button("🗑 Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# -------------------------------------------------------
# Main Title
# -------------------------------------------------------

st.title("⚡ Schneider Electric ACB AI Assistant")

st.caption(
    "Ask questions about MasterPacT MTZ documentation."
)

# -------------------------------------------------------
# Display Chat History
# -------------------------------------------------------

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])

        if message["role"] == "assistant":

            sources = message.get("sources", [])

            if sources:

                with st.expander("Sources"):

                    for src in sources:
                        st.write(f"• {src}")

# -------------------------------------------------------
# User Input
# -------------------------------------------------------

prompt = st.chat_input(
    "Ask about MasterPacT MTZ..."
)

if prompt:

    st.session_state.messages.append(
        {
            "role": "user",
            "content": prompt,
        }
    )

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):

        with st.spinner("Searching documentation..."):

            response = st.session_state.chatbot.ask(prompt)

            if isinstance(response, dict):

                answer = response["answer"]

                sources = response.get("sources", [])

            else:

                answer = response

                sources = []

        st.markdown(answer)

        if sources:

            with st.expander("Sources"):

                for src in sources:

                    st.write(f"• {src}")

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
            "sources": sources,
        }
    )