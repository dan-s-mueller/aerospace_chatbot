import streamlit as st
import os, time
from streamlit_pdf_viewer import pdf_viewer
import tempfile

from aerospace_chatbot.ui import SidebarManager, display_chat_history, display_sources
from aerospace_chatbot.processing import QAModel, DocumentProcessor
from aerospace_chatbot.services import EmbeddingService, LLMService, DatabaseService
from aerospace_chatbot.core import get_secrets

def handle_file_upload(sb, secrets):
    """Handle file upload functionality for the chatbot."""
    if not _validate_upload_settings(sb):
        return

    uploaded_files = st.file_uploader("Choose pdf files", accept_multiple_files=True)
    if not uploaded_files:
        return

    temp_files = _save_uploads_to_temp(uploaded_files)
    
    if st.button('Upload your docs into vector database'):
        with st.spinner('Uploading and merging your documents with the database...'):
            _process_uploads(sb, secrets, temp_files)

    if st.session_state.user_upload:
        st.markdown(
            f":white_check_mark: Merged! Your upload ID: `{st.session_state.user_upload}`. "
            "This will be used for this chat session to also include your documents. "
            "When you restart the chat, you'll have to re-upload your documents."
        )
def _validate_upload_settings(sb):
    """Validate RAG type and index type settings."""
    if sb['rag_type'] != "Standard":
        st.error("Only Standard RAG is supported for user document upload.")
        return False
    if sb['index_type'] != 'Pinecone':
        st.error("Only Pinecone is supported for user document upload.")
        return False
    
    st.write("Upload parameters set to standard values, hard coded for now...standard only")
    return True
def _save_uploads_to_temp(uploaded_files):
    """Save uploaded files to temporary directory."""
    temp_files = []
    for uploaded_file in uploaded_files:
        temp_path = os.path.join(tempfile.gettempdir(), uploaded_file.name)
        with open(temp_path, 'wb') as temp_file:
            temp_file.write(uploaded_file.read())
        temp_files.append(temp_path)
    return temp_files
def _process_uploads(sb, secrets, temp_files):
    """Process uploaded files and merge them into the vector database."""
    # Initialize services
    db_service = DatabaseService(
        db_type=sb['index_type'],
        local_db_path=os.getenv('LOCAL_DB_PATH')
    )
    embedding_service = EmbeddingService(
        model_name=sb['embedding_name'],
        model_type=sb['query_model'],
        api_key=secrets.get(f"{sb['query_model']}_key")
    )
    # Initialize document processor
    doc_processor = DocumentProcessor(
        db_service=db_service,
        embedding_service=embedding_service
    )
    
    # Generate unique identifier
    st.session_state.user_upload = f"user_upload_{os.urandom(3).hex()}"
    st.markdown("*Processing and uploading user documents...*")
    
    # Process and index documents
    st.markdown("*Uploading user documents to namespace...*")
    chunking_result = doc_processor.process_documents(temp_files)
    doc_processor.index_documents(
        index_name=sb['index_selected'],
        chunking_result=chunking_result,
        namespace=st.session_state.user_upload,
        show_progress=True
    )
    # Copy vectors to merge namespaces
    st.markdown("*Merging user document with existing documents...*")
    doc_processor.copy_vectors(
        index_name=sb['index_selected'],
        source_namespace=None,
        target_namespace=st.session_state.user_upload,
        show_progress=True
    )
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
st.title('🚀 Aerospace Chatbot')
current_directory = os.path.dirname(os.path.abspath(__file__))
home_dir = os.path.abspath(os.path.join(current_directory, "../../"))

# Initialize SidebarManager
sidebar_manager = SidebarManager(st.session_state.config_file)
sb = sidebar_manager.render_sidebar()
secrets = get_secrets()

# Disable sidebar elements after first message
if 'message_id' in st.session_state and st.session_state.message_id > 0:
    # Disable individual sidebar elements
    st.session_state['index_disabled'] = True
    st.session_state['embeddings_disabled'] = True
    st.session_state['rag_disabled'] = True
    st.session_state['llm_disabled'] = True
    st.session_state['model_options_disabled'] = True

# Set up chat history
if 'user_upload' not in st.session_state:
    st.session_state.user_upload = None
if 'qa_model_obj' not in st.session_state:
    st.session_state.qa_model_obj = None
if 'message_id' not in st.session_state:
    st.session_state.message_id = 0
if 'messages' not in st.session_state:
    st.session_state.messages = []
display_chat_history(st.session_state.messages)
if 'pdf_urls' not in st.session_state:
    st.session_state.pdf_urls = []
reset_query_db=False    # Add reset option for query database

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

if st.session_state.message_id == 0:
    with st.expander("Upload files to existing database",expanded=True):
        handle_file_upload(sb, secrets)

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
            t_start = time.time()
            st.session_state.message_id += 1
            st.write(f'*Starting response generation for message: {str(st.session_state.message_id)}*')
            
            # Initialize embedding service
            embedding_service = EmbeddingService(
                model_name=sb['embedding_name'],
                model_type=sb['query_model'],
                api_key=secrets.get(f"{sb['query_model']}_key")
            )
            
            # Initialize LLM service
            llm_service = LLMService(
                model_name=sb['llm_model'],
                model_type=sb['llm_source'],
                api_key=secrets.get(f"{sb['llm_source']}_key"),
                temperature=sb['model_options']['temperature'],
                max_tokens=sb['model_options']['output_level']
            )
            
            # Initialize services, initialie database
            db_service = DatabaseService(
                db_type=sb['index_type'],
                local_db_path=os.getenv('LOCAL_DB_PATH')
            )
            db_service.initialize_database(
                index_name=sb['index_selected'],
                embedding_service=embedding_service,
                rag_type=sb['rag_type'],
                namespace=st.session_state.user_upload
            )

            # Initialize QA model
            st.session_state.qa_model_obj = QAModel(
                db_service=db_service,
                llm_service=llm_service,
                k=sb['model_options']['k'],
                namespace=st.session_state.user_upload
            )

            st.write('*Searching vector database, generating prompt...*')
            result = st.session_state.qa_model_obj.query(prompt)
            ai_response = st.session_state.qa_model_obj.ai_response

            message_placeholder.markdown(ai_response)
            similar_questions = st.session_state.qa_model_obj.generate_similar_questions(prompt)
            st.info("**Alternative questions:**\n\n" + "\n".join(similar_questions))

            t_delta = time.time() - t_start
            status.update(
                label=':white_check_mark: Prompt generated in '+"{:10.3f}".format(t_delta)+' seconds', 
                state='complete', 
                expanded=False
            )
            
            # Display sources
            display_sources(st.session_state.qa_model_obj.sources[-1])

            st.session_state.messages.append({'role': 'assistant', 'content': ai_response})

# Add reset button
if st.button('Restart session'):
    _reset_conversation()
