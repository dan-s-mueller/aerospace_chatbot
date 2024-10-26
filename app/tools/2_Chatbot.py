import os, sys, time, ast
import streamlit as st

from langchain_community.vectorstores import Pinecone
# from langchain_community.vectorstores import Chroma
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_voyageai import VoyageAIEmbeddings
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings

import nltk # Do before ragatioulle import to avoid logs
nltk.download('punkt', quiet=True)
from ragatouille import RAGPretrainedModel

sys.path.append('../../src/aerospace_chatbot')  # Add package to path
import admin, queries

import streamlit as st
import fitz  # PyMuPDF
import requests
import base64
from streamlit_pdf_viewer import pdf_viewer

def _reset_conversation():
    """
    Resets the conversation by clearing the session state variables related to the chatbot.

    Returns:
        None
    """
    st.session_state.qa_model_obj = []
    st.session_state.message_id = 0
    st.session_state.messages = []
    st.session_state.pdf_urls = []
    return None

def extract_pages_from_pdf(url, target_page, page_range=5):
    # Download extracted relevant section of the PDF file
    response = requests.get(url)
    pdf_data = response.content

    # Load PDF in PyMuPDF
    doc = fitz.open("pdf", pdf_data)
    extracted_doc = fitz.open()  # New PDF for extracted pages

    # Calculate the range of pages to extract
    start_page = max(target_page - page_range, 0)
    end_page = min(target_page + page_range, doc.page_count - 1)

    # Extract specified pages
    for i in range(start_page, end_page + 1):
        extracted_doc.insert_pdf(doc, from_page=i, to_page=i)

    # Save the extracted pages to a new PDF file in memory
    extracted_pdf = extracted_doc.tobytes()
    extracted_doc.close()
    doc.close()
    return extracted_pdf

def get_pdf(url):
    # Download full PDF file
    response = requests.get(url)
    pdf_data = response.content

    # Load PDF in PyMuPDF
    doc = fitz.open("pdf", pdf_data)

    # Save the extracted pages to a new PDF file in memory
    extracted_pdf = doc.tobytes()
    doc.close()
    return extracted_pdf

# Page setup
current_directory = os.path.dirname(os.path.abspath(__file__))
home_dir = os.path.abspath(os.path.join(current_directory, "../../"))
paths,sb,secrets=admin.st_setup_page('Aerospace Chatbot',
                                     home_dir,
                                     st.session_state.config_file,
                                     {'vector_database':True,
                                      'embeddings':True,
                                      'rag_type':True,
                                      'index_selected':True,
                                      'llm':True,
                                      'model_options':True,
                                      'secret_keys':True})

# Add expander with functionality details.
with st.expander('''What's under the hood?'''):
    st.markdown('''
    This chatbot will look up from all Aerospace Mechanism Symposia in the following location: https://github.com/dan-s-mueller/aerospace_chatbot/tree/main/data/AMS
    
    A long list of example questions for the Aerospace Mechanisms chatbot, with other details is located here: https://aerospace-chatbot.readthedocs.io/en/latest/help/deployments.html
    ''')

# Add reset option for query database
reset_query_db = st.empty()
reset_query_db.checkbox('Reset query database?', value=False, help='This will reset the query database used for visualization.')

# Set up chat history
if 'qa_model_obj' not in st.session_state:
    st.session_state.qa_model_obj = []
if 'message_id' not in st.session_state:
    st.session_state.message_id = 0
if 'messages' not in st.session_state:
    st.session_state.messages = []
for message in st.session_state.messages:
    with st.chat_message(message['role']):
        st.markdown(message['content'])

# Define chat
if prompt := st.chat_input('Prompt here'):
    # User prompt
    st.session_state.messages.append({'role': 'user', 'content': prompt})
    with st.chat_message('user'):
        st.markdown(prompt)
    # Assistant response
    with st.chat_message('assistant'):
        message_placeholder = st.empty()

        with st.status('Generating response...') as status:
            t_start=time.time()

            st.session_state.message_id += 1
            st.write(f'*Starting response generation for message: {str(st.session_state.message_id)}*')
            
            if st.session_state.message_id==1:  # Initialize chat
                query_model = admin.get_query_model(sb, secrets)    # Set query model
                llm=admin.set_llm(sb,secrets,type='prompt') # Define LLM

                # Initialize QA model object
                if 'search_type' in sb['model_options']: 
                    search_type=sb['model_options']['search_type']
                else:
                    search_type=None
                st.session_state.qa_model_obj=queries.QA_Model(sb['index_type'],
                                                               sb['index_selected'],
                                                               query_model,
                                                               llm,
                                                               rag_type=sb['rag_type'],
                                                               k=sb['model_options']['k'],
                                                               search_type=search_type,
                                                               local_db_path=paths['db_folder_path'],
                                                               reset_query_db=reset_query_db)
                reset_query_db.empty()  # Remove this option after initialization
            if st.session_state.message_id>1:   # Chat after first message and initialization
                # Update LLM
                llm=admin.set_llm(sb,secrets,type='prompt')
                st.session_state.qa_model_obj.update_model(llm)
            
            st.write('*Searching vector database, generating prompt...*')
            st.write(f"*Query added to query database: {sb['index_selected']+'-queries'}*")
            st.session_state.qa_model_obj.query_docs(prompt)
            ai_response=st.session_state.qa_model_obj.ai_response

            message_placeholder.markdown(ai_response)
            st.info("**Alternative questions:** \n\n\n"+
                     st.session_state.qa_model_obj.generate_alternative_questions(prompt))

            t_delta=time.time() - t_start
            status.update(label='Prompt generated in '+"{:10.3f}".format(t_delta)+' seconds', state='complete', expanded=False)
            
        # Create a dropdown box with hyperlinks to PDFs and their pages
        if st.session_state.qa_model_obj.sources[-1]:
            with st.expander("View Source Documents",expanded=True):
                pdf_urls = []
                for source in st.session_state.qa_model_obj.sources[-1]:
                    page = source.get('page')
                    pdf_source = source.get('source')
                    
                    # Parse string representations of lists
                    try:
                        page = ast.literal_eval(page) if isinstance(page, str) else page
                        pdf_source = ast.literal_eval(pdf_source) if isinstance(pdf_source, str) else pdf_source
                    except (ValueError, SyntaxError):
                        # If parsing fails, keep the original value
                        pass

                    # Extract first element if it's a list
                    page = page[0] if isinstance(page, list) and page else page
                    pdf_source = pdf_source[0] if isinstance(pdf_source, list) and pdf_source else pdf_source
                    
                    if pdf_source and page is not None:
                        pdf_urls.append(f"https://storage.googleapis.com/ams-chatbot-pdfs/{pdf_source}")
                        st.markdown(f"- [{pdf_source}](https://storage.googleapis.com/ams-chatbot-pdfs/{pdf_source}) - Page {page}")
            # Store pdf_urls in session state
            st.session_state.pdf_urls = pdf_urls

        st.session_state.messages.append({'role': 'assistant', 'content': ai_response})

if 'pdf_urls' in st.session_state and st.session_state.pdf_urls:
    # Page viewer
    with st.expander("View Source Documents", expanded=True):
        # Example PDF selection and page input
        selected_url = st.selectbox("Select a PDF to preview:", st.session_state.pdf_urls)
        target_page = st.number_input("Enter the target page:", min_value=1, value=10) - 1

        # Extract and display the pages when the user clicks
        if st.button("Show Pages"):
            full_pdf = get_pdf(selected_url)
            extracted_pdf = extract_pages_from_pdf(selected_url, target_page)

            # Display or download the PDF
            st.download_button("Download Extracted Pages", data=extracted_pdf, file_name="extracted_pages.pdf")
            
            st.write("Here are your extracted pages:")
            pdf_viewer(extracted_pdf,
                    width=1000,
                    height=1200,
                    scroll_to_page=5)
            
            st.write("Here is the full PDF:")
            pdf_viewer(full_pdf,
                    width=1000,
                    height=1200,
                    scroll_to_page=target_page)


# Add reset button
if st.button('Restart session'):
    _reset_conversation()
