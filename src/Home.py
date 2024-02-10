import admin

import streamlit as st
import os

# Set up page
st.set_page_config(
    page_title="Aerospace Chatbot: AMS",
)
st.title("Aerospace Chatbot Homepage")
st.subheader("Aerospace Mechanisms Symposia (AMS)")
st.markdown("""
This space contains chatbots and tools for exploring data in the aerospace mechanisms symposia, using all available papers published since 2000.
Those papers are located here: https://huggingface.co/spaces/ai-aerospace/aerospace_chatbots/tree/main/data/AMS
""")
st.subheader("Code Details")
st.markdown("Code base: https://github.com/dsmueller3760/aerospace_chatbot/tree/rag_study")
st.markdown(
    '''
    API key links:
    * OpenAI: https://platform.openai.com/api-keys
    * Pinecone: https://www.pinecone.io
    * Hugging Face: https://huggingface.co/settings/tokens
    * Voyage: https://dash.voyageai.com/api-keys
    ''')
with st.expander("Connection Status",expanded=True):
    st.markdown("**API keys** (Indicates status of local variable. It does not guarantee the key itself is correct):")
    st.markdown(admin.test_key_status())
    st.markdown(admin.show_pinecone_indexes())
    st.markdown(admin.show_chroma_collections())
    st.markdown(admin.test_ragatouille_status())

st.subheader("Running Locally")
'''
It is recommended to run this streamlit app locally for improved performance. The hosted hugging face version is for proof of concept.
You must have poetry installed locally to manage depdenencies. To run locally, clone the repository and run the following commands.
    
    poetry config virtualenvs.in-project true
    poetry install
    source .venv/bin/activate
    cd ./src
    streamlit run Home.py
'''