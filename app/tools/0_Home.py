import streamlit as st
import sys, os


sys.path.append('../src/aerospace_chatbot')   # Add package to path
import admin

# Set up page
current_directory = os.path.dirname(os.path.abspath(__file__))
home_dir = os.path.abspath(os.path.join(current_directory, "../../"))
paths,sb,secrets=admin.st_setup_page('Aerospace Chatbot',
                                     home_dir,
                                     st.session_state.config_file)

st.info("""
        * [Help Docs](https://aerospace-chatbot.readthedocs.io/en/latest/index.html)
        * [Code Repository](https://github.com/dan-s-mueller/aerospace_chatbot)
        * For questions and problem reporting, please create an issue [here](https://github.com/dan-s-mueller/aerospace_chatbot/issues/new)
        """)

# Two run modes: admin and tester. Tester has limited functionality for sharing to a broad audience. Admin has full functionality.
user=os.getenv('AEROSPACE_CHATBOT_CONFIG')
if user=="admin":
    st.subheader("Aerospace Mechanisms Chatbot")
    st.markdown("""
    This tool was developed based on all [Aerospace Mecahnisms Symposia](https://aeromechanisms.com/past-symposia/) papers published since 2000. 
    Use of this tool using other data sources is possible but hasn't been extensively tested.
    Each paper has had Optical Character Recognition (OCR) reportofmed using the latest release of [OCR my PDF](https://ocrmypdf.readthedocs.io/en/latest/).

    All of these papers are available in the following location: https://github.com/dan-s-mueller/aerospace_chatbot/tree/main/data/AMS
    """)

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
elif user=="tester":
    st.subheader("Aerospace Mechanisms Chatbot")
    st.markdown("""
    This is a beta tester vesion of the Aerospace Chatbot, deployed with expertise in Aerospace Mechanisms.
    The tool comes loaded with all [Aerospace Mecahnisms Symposia](https://aeromechanisms.com/past-symposia/) papers published since 2000. 
    Each paper has had Optical Character Recognition (OCR) reperformend using the latest release of [OCR my PDF](https://ocrmypdf.readthedocs.io/en/latest/).

    All of these papers are available in the following location: https://github.com/dan-s-mueller/aerospace_chatbot/tree/main/data/AMS
    """)