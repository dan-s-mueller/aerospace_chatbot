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
Chatbots for aerospace mechanisms symposia, using all available papers published since 2000
* Modular version meant to study retrieval methods
""")
st.subheader("AMS")
'''
This chatbot will look up from all Aerospace Mechanism Symposia in the following location: https://github.com/dsmueller3760/aerospace_chatbot/tree/main/data/AMS
* Available models: https://platform.openai.com/docs/models
* Model parameters: https://platform.openai.com/docs/api-reference/chat/create
* Pinecone: https://docs.pinecone.io/docs/projects#api-keys
* OpenAI API: https://platform.openai.com/api-keys
'''

# # Establish secrets
# PINECONE_ENVIRONMENT=os.getenv('PINECONE_ENVIRONMENT')
# PINECONE_API_KEY=os.getenv('PINECONE_API_KEY')