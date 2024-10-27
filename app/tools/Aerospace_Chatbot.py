import os, sys, time, ast
import streamlit as st
from streamlit_pdf_viewer import pdf_viewer
import tempfile
import chromadb

sys.path.append('../src/aerospace_chatbot')   # Add package to path
import admin, queries, data_processing

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

# Page setup
current_directory = os.path.dirname(os.path.abspath(__file__))
home_dir = os.path.abspath(os.path.join(current_directory, "../../"))
paths,sb,secrets=admin.st_setup_page('🚀 Aerospace Chatbot',
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
with st.expander('''Helpful Information'''):
    st.info("""
            * [Help Docs](https://aerospace-chatbot.readthedocs.io/en/latest/index.html)
            * [Code Repository](https://github.com/dan-s-mueller/aerospace_chatbot)
            * For questions and problem reporting, please create an issue [here](https://github.com/dan-s-mueller/aerospace_chatbot/issues/new)
            """)
    st.subheader("Aerospace Mechanisms Chatbot")
    st.markdown("""
    This is a beta version vesion of the Aerospace Chatbot. The tool comes loaded with a subset of the [Aerospace Mecahnisms Symposia](https://aeromechanisms.com/past-symposia/) papers. To view the latest status of what papers are included, please see the [Aerospace Chatbot Documents Library](https://docs.google.com/spreadsheets/d/1Fv_QGENr2W8Mh_e-TmoWagkhpv7IImw3y3_o_pJcvdY/edit?usp=sharing)
    
    To enable optimal retrieval, each paper has had Optical Character Recognition (OCR) reperformend using the latest release of [OCR my PDF](https://ocrmypdf.readthedocs.io/en/latest/).

    """)


with st.expander("Upload files to existing database",expanded=True):
    if sb['rag_type']=="Standard":
        st.write("Upload parameters set to standard values, hard coded for now...standard only")

        uploaded_files = st.file_uploader(
            "Choose pdf files", accept_multiple_files=True
        )
        temp_files=[]
        for uploaded_file in uploaded_files:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                    temp_file.write(uploaded_file.read())
                    temp_files.append(temp_file.name)

        # Retrieve the query model from the selected Chroma database
        chroma_client = chromadb.PersistentClient(path=os.path.join(paths['db_folder_path'],'chromadb'))
        selected_collection = chroma_client.get_collection(sb['index_selected'])
        query_model = admin.get_query_model({'index_type':sb['index_type'],
                                             'query_model':selected_collection.metadata['query_model'],
                                             'embedding_name':selected_collection.metadata['embedding_model']},
                                             {'OPENAI_API_KEY':os.getenv('OPENAI_API_KEY')})

        # Upload documents to vector database selected
        # Get subset of chunker parameters, can't store in own dict due to ChromaDB metadata limitations
        chunk_params = {
            key: value for key, value in selected_collection.metadata.items() 
            if key not in ['query_model', 'embedding_model'] and value is not None    
        }
        if st.button('Upload your docs into vector database'):
            data_processing.load_docs(sb['index_type'],
                            temp_files,
                            query_model,
                            index_name=sb['index_selected'],
                            local_db_path=paths['db_folder_path'],
                            show_progress=True,
                            **chunk_params)
    else:
        st.error("Only Standard RAG is supported for user document upload.")

# Add reset option for query database
reset_query_db=False
# reset_query_db = st.empty()
# reset_query_db.checkbox('Reset query database?', value=False, help='This will reset the query database used for visualization.')

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
if 'pdf_urls' not in st.session_state:
    st.session_state.pdf_urls = []

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
                # reset_query_db.empty()  # Remove this option after initialization
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
            status.update(label=':white_check_mark: Prompt generated in '+"{:10.3f}".format(t_delta)+' seconds', state='complete', expanded=False)
            
        # Create a dropdown box with hyperlinks to PDFs and their pages
        with st.container():
            with st.spinner('Bring you source documents...'):
                st.write(":notebook: Source Documents")
                for source in st.session_state.qa_model_obj.sources[-1]:
                    st.session_state.pdf_urls=[]
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
                        selected_url = f"https://storage.googleapis.com/ams-chatbot-pdfs/{pdf_source}"
                        st.session_state.pdf_urls.append(selected_url)
                        st.markdown(f"[{pdf_source} (Download)]({selected_url}) - Page {page}")

                        with st.expander(":memo: View"):
                            tab1, tab2 = st.tabs(["Relevant Context+5 Pages", "Full"])
                            try:
                                # Extract and display the pages when the user clicks
                                extracted_pdf = admin.extract_pages_from_pdf(selected_url, page)
                                with tab1:
                                    pdf_viewer(extracted_pdf,width=1000,height=1200,render_text=True)
                                with tab2:
                                    # pdf_viewer(full_pdf, width=1000,height=1200,render_text=True)
                                    st.write("Disabled for now...see download link above!")
                            except Exception as e:
                                st.warning("Unable to load PDF preview. The file may no longer exist or be inaccessible. Contact support if this issue persists.")

        st.session_state.messages.append({'role': 'assistant', 'content': ai_response})

# Add reset button
if st.button('Restart session'):
    _reset_conversation()
