import admin
import data_processing

import streamlit as st
import os

import logging
from dotenv import load_dotenv,find_dotenv

# Set up the page, enable logging, read environment variables
load_dotenv(find_dotenv(),override=True)
# logging.basicConfig(filename='app_Home.log', filemode='w', format='%(name)s - %(levelname)s - %(message)s', level=logging.DEBUG)

# Set up page
st.set_page_config(
    page_title="Aerospace Chatbot",
)
st.title("Aerospace Chatbot Homepage")
st.subheader("Aerospace Mechanisms Symposia (AMS)")
st.markdown("""
This space contains chatbots and tools for exploring data in the aerospace mechanisms symposia, using all available papers published since 2000.
Those papers are located here: https://github.com/dan-s-mueller/aerospace_chatbot/tree/main/data/AMS
""")
st.subheader("Help Docs")
st.markdown("https://aerospace-chatbot.readthedocs.io/en/latest/index.html")
st.subheader("Running Locally")
st.markdown(
    '''
    It is recommended to run this streamlit app locally for improved performance. See here for details: https://aerospace-chatbot.readthedocs.io/en/latest/index.html#running-locally
    ''')
st.subheader("Code Details")
st.markdown("Code base: https://github.com/dan-s-mueller/aerospace_chatbot")
st.markdown(
    '''
    API key links:
    * OpenAI: https://platform.openai.com/api-keys
    * Pinecone: https://www.pinecone.io
    * Hugging Face: https://huggingface.co/settings/tokens
    * Voyage: https://dash.voyageai.com/api-keys
    ''')

st.subheader('Connection Status')
st.markdown('Vector database deletion possible through Database Processing page with account.')
admin.st_connection_status_expander(delete_buttons=False)