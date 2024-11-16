import streamlit as st
import os, time

from aerospace_chatbot.core.config import setup_logging
from aerospace_chatbot.ui import SidebarManager, display_chat_history, display_sources, handle_file_upload
from aerospace_chatbot.processing import QAModel
from aerospace_chatbot.services import EmbeddingService, LLMService, DatabaseService

logger = setup_logging()

# Initialize session state variables if they don't exist
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'message_id' not in st.session_state:
    st.session_state.message_id = 0
if 'qa_model_obj' not in st.session_state:
    st.session_state.qa_model_obj = None
if 'pdf_urls' not in st.session_state:
    st.session_state.pdf_urls = []
if 'user_upload' not in st.session_state:
    st.session_state.user_upload = None

def _reset_conversation():
    """Resets the conversation by clearing the session state variables related to the chatbot."""
    if 'qa_model_obj' in st.session_state:
        # Clean up existing database connections
        if st.session_state.qa_model_obj:
            try:
                st.session_state.qa_model_obj.query_db_service.delete_index()
            except:
                pass
    
    st.session_state.qa_model_obj = None
    st.session_state.message_id = 0
    st.session_state.messages = []
    st.session_state.pdf_urls = []
    st.session_state.user_upload = None
    return None

# Page setup
# st.title('🚀 Aerospace Chatbot')

# Create fixed containers
header = st.container()
info_section = st.container()
upload_section = st.container()
chat_section = st.container()

# Initialize SidebarManager first (since other components might depend on it)
sidebar_manager = SidebarManager(st.session_state.config_file)
sb = sidebar_manager.render_sidebar()

# Header section
with header:
    st.title('🚀 Aerospace Chatbot')

# Info section - Always visible at top
with info_section:
    with st.expander('Helpful Information'):
        st.info("""
                * [Help Docs](https://aerospace-chatbot.readthedocs.io/en/latest/index.html)
                * [Code Repository](https://github.com/dan-s-mueller/aerospace_chatbot)
                * For questions and problem reporting, please create an issue [here](https://github.com/dan-s-mueller/aerospace_chatbot/issues/new)
                """)
        st.subheader("Aerospace Mechanisms Chatbot")
        st.markdown("""
        This is a beta version of the Aerospace Chatbot. The tool comes loaded with a subset of the [Aerospace Mechanisms Symposia](https://aeromechanisms.com/past-symposia/) papers. To view the latest status of what papers are included, please see the [Aerospace Chatbot Documents Library](https://docs.google.com/spreadsheets/d/1Fv_QGENr2W8Mh_e-TmoWagkhpv7IImw3y3_o_pJcvdY/edit?usp=sharing)
        
        To enable optimal retrieval, each paper has had Optical Character Recognition (OCR) reperformed using the latest release of [OCR my PDF](https://ocrmypdf.readthedocs.io/en/latest/).
        """)

# Upload section - Only visible when no messages
with upload_section:
    if len(st.session_state.get('messages', [])) == 0:
        with st.expander("Upload files to existing database", expanded=True):
            st.session_state.user_upload = handle_file_upload(sb)

# Chat section
with chat_section:
    chat_col, sources_col = st.columns([1, 1])

    # Left column for chat
    with chat_col:
        # Check if there's a selected question from button click
        if 'selected_question' in st.session_state:
            prompt = st.session_state.selected_question
            # Clear it so it doesn't trigger again
            del st.session_state.selected_question
        else:
            # Regular chat input
            prompt = st.chat_input("Prompt here")
        if st.button('Restart session'):
            _reset_conversation()
            st.rerun()
        
        # If there's a new prompt, process it first
        if prompt:
            # Show processing message first
            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                try:
                    t_start = time.time()
                    with st.status('Generating response...') as status:
                        st.write(f'*Starting response generation for message: {str(st.session_state.message_id)}*')
                        st.write(f'*Prompt: {prompt}*')
                        
                        # Initialize services
                        embedding_service = EmbeddingService(
                            model_name=sb['embedding_name'],
                            model_type=sb['embedding_model']
                        )
                        
                        llm_service = LLMService(
                            model_name=sb['llm_model'],
                            model_type=sb['llm_source'],
                            temperature=sb['model_options']['temperature'],
                            max_tokens=sb['model_options']['output_level']
                        )
                        
                        # Initialize database service
                        db_service = DatabaseService(
                            db_type=sb['index_type'],
                            index_name=sb['index_selected'],
                            rag_type=sb['rag_type'],
                            embedding_service=embedding_service,
                            doc_type='document'
                        )
                        
                        try:
                            db_service.initialize_database(
                                namespace=st.session_state.user_upload
                            )
                        except ValueError as e:
                            st.error(f"Database initialization failed: {str(e)}")
                            st.stop()

                        # Initialize QA model
                        if not st.session_state.qa_model_obj:
                            st.session_state.qa_model_obj = QAModel(
                                db_service=db_service,
                                llm_service=llm_service,
                                k=sb['model_options']['k'],
                                namespace=st.session_state.user_upload
                            )

                        st.write('*Searching vector database, generating prompt...*')
                        result = st.session_state.qa_model_obj.query(prompt)
                        ai_response = st.session_state.qa_model_obj.ai_response
                        similar_questions = st.session_state.qa_model_obj.generate_alternative_questions(prompt)
                        
                        message_placeholder.markdown(ai_response)
                        response_time = time.time() - t_start
                        
                        # Add messages to session state in correct order
                        st.session_state.messages.insert(0, {
                            'role': 'assistant', 
                            'content': ai_response, 
                            'sources': st.session_state.qa_model_obj.sources[-1],
                            'alternative_questions': similar_questions,
                            'response_time': response_time,
                            'message_id': st.session_state.message_id
                        })
                        st.session_state.messages.insert(1, {"role": "user", "content": prompt})
                        st.rerun()
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")
                    _reset_conversation()
                    st.stop()
        
        # Always show message history
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                if message["role"] == "assistant":
                    with st.status('Response Status', state="complete") as status:
                        st.write(f'Message ID: {message.get("message_id", "N/A")}')
                        st.write('✅ Response generated successfully')
                        if 'response_time' in message:
                            st.write(f'⏱️ Response time: {message["response_time"]:.2f} seconds')

    # Right column - Persistent display of last message's info
    with sources_col:
        if st.session_state.messages and len(st.session_state.messages) > 0:
            last_message = st.session_state.messages[0]  # Get most recent message
            
            # Alternative questions section at the top
            with st.container():
                if 'alternative_questions' in last_message:
                    st.markdown("💡 Alternative Questions", help="Click on a question to use as a prompt.")
                    # Replace st.info with buttons for each question
                    for question in last_message['alternative_questions']:
                        if st.button(f"🔄 {question}", key=f"btn_{hash(question)}"):
                            # Store the question in session state
                            st.session_state.selected_question = question
                            st.rerun()
            
            # Source documents section below
            with st.container():
                if 'sources' in last_message:
                    st.markdown("📚 Source Documents")
                    display_sources(last_message['sources'])

        # If we're processing a new message, show its info too
        if prompt:
            with st.spinner('Loading response details...'):
                # This will show while the response is being generated
                if 'qa_model_obj' in st.session_state and st.session_state.qa_model_obj:
                    if hasattr(st.session_state.qa_model_obj, 'sources'):
                        display_sources(st.session_state.qa_model_obj.sources[-1])
