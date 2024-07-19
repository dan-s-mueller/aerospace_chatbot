import streamlit as st
import sys, os


sys.path.append('../src/aerospace_chatbot')   # Add package to path
import admin

# Set up page, enable logging
current_directory = os.path.dirname(os.path.abspath(__file__))
home_dir = os.path.abspath(os.path.join(current_directory, "../../"))
paths,sb,secrets=admin.st_setup_page('Aerospace Chatbot Homepage',
                                     home_dir)

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
admin.st_connection_status_expander(delete_buttons=False,set_secrets=True)