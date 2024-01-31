import streamlit as st
import os

# Set up page
st.set_page_config(
    page_title="Aerospace Chatbot: AMS",
)
st.title("Aerospace Chatbot Homepage")
st.markdown("Code base: https://github.com/dsmueller3760/aerospace_chatbot/tree/rag_study")
st.markdown('---')
st.title("Chatbots")
st.markdown("""
This space contains chatbots and tools for exploring data in the aerospace mechanisms symposia, using all available papers published since 2000.
""")
st.subheader("Running Locally")
'''
It is recommended to run this streamlit app locally for improved performance. The hosted hugging face version is for proof of concept.
You must have poetry installed locally to manage depdenencies. To run locally, clone the repository and run the following commands.
    
    poetry config virtualenvs.in-project true
    poetry install
    source .venv/bin/activate
    cd ./scripts
    streamlit run Start.py
'''

st.subheader("Aerospace Mechanisms Symposia (AMS)")
'''
This chatbot will look up from all Aerospace Mechanism Symposia in the following location: https://github.com/dsmueller3760/aerospace_chatbot/tree/main/data/AMS
* Available models: https://platform.openai.com/docs/models
* Model parameters: https://platform.openai.com/docs/api-reference/chat/create
* Pinecone: https://docs.pinecone.io/docs/projects#api-keys
* OpenAI API: https://platform.openai.com/api-keys
'''