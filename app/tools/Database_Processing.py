import os, sys, time
# from langchain_core.documents import Document
import streamlit as st

from aerospace_chatbot.ui import SidebarManager, show_connection_status, handle_sidebar_state
from aerospace_chatbot.processing import DocumentProcessor
from aerospace_chatbot.services import (
    EmbeddingService, 
    LLMService, 
    DatabaseService, 
    get_available_indexes
)

# Page setup
st.title('📓 Database Processing')

# Initialize SidebarManager first
if 'sidebar_state_initialized' not in st.session_state:
    st.session_state.sidebar_manager = SidebarManager(st.session_state.config_file)
    st.session_state.sb = {}

# Handle sidebar state
st.session_state.sb = handle_sidebar_state(st.session_state.sidebar_manager)

# Page setup
st.subheader('Connection status and vector database cleanup')
show_connection_status(expanded=False, delete_buttons=True)

# Add section for creating and loading into a vector database
st.subheader('Create and load into a vector database')

# Initialize services after potential rerun
embedding_service = EmbeddingService(
    model_service=st.session_state.sb['embedding_service'],
    model=st.session_state.sb['embedding_model']
)

# Find docs
# TODO add option for local documents
st.session_state.buckets = None
try:
    st.session_state.buckets = DocumentProcessor.list_available_buckets()
    
    if not st.session_state.buckets:
        st.warning("No Google Cloud Storage buckets found. Please ensure you have access to at least one bucket in your Google Cloud project.")
        st.stop()
        
    bucket_name = st.selectbox('Select Google Cloud Storage bucket',
                             options=st.session_state.buckets,
                             help='Select a Google Cloud Storage bucket containing PDFs')

    docs = DocumentProcessor.list_bucket_pdfs(bucket_name)
    markdown_text = f"**Number of PDFs found:** {len(docs)}\n"
    if len(docs) > 0:
        markdown_text += "**PDFs found:**\n"
        for doc in docs:
            filename = doc.split('/')[-1]
            markdown_text += f"- `{filename}`\n"
    else:
        st.info(f"No PDF files found in bucket '{bucket_name}'")
    st.markdown(markdown_text)
except Exception as e:
    st.error(f'Error accessing GCS: {str(e)}. Please check your credentials and permissions. You may need to log into your Google Cloud account (`gcloud auth login`), or access to the bucket.')

# Set database name
index_appendix = st.text_input('Appendix for index name', 'mch')
st.session_state.index_name = (st.session_state.sb['embedding_model'].replace('/', '-').replace(' ', '-') + '-' + index_appendix).lower()

# Add an expandable box for options
with st.expander("Options",expanded=True):
    clear_database = st.checkbox('Delete existing database?',value=True,help='If checked, the existing database will be deleted. New databases will have metadata added to the index for chunking and embedding model.')
    partition_by_api=st.checkbox('Partition by API?',value=False,help='If checked, documents will be partitioned by the API instead of locally.')
    batch_size=st.number_input('Batch size for upsert', 
                    min_value=1, max_value=1000, step=1, value=500,
                    help='''The number of documents to upsert at a time. 
                            Useful for hosted databases (e.g. Pinecone), or those that require long processing times.
                            When using hugging face embeddings without a dedicated endpoint, batch size recommmended maximum is 32.''')
    chunk_size=st.number_input('Chunk size (tokens)', min_value=1, step=1, value=400, help='Token dependent on model.')
    chunk_overlap=st.number_input('Chunk overlap (tokens)', min_value=0, step=1, value=0)

# Initialize database service
db_service = DatabaseService(
    db_type=st.session_state.sb['index_type'],
    index_name=st.session_state.index_name,
    embedding_service=embedding_service
)

# Add a button to run the function
if st.button('Load docs into vector database'):
    with st.spinner('Processing and loading documents into vector database...'):
        start_time = time.time()

        # Initialize document processor
        doc_processor = DocumentProcessor(
            embedding_service=embedding_service,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            work_dir='../data/document_processing'
        )

        try:
            # Initialize database
            db_service.initialize_database(clear=clear_database)
            
            # Process documents (partition and chunk)
            partitioned_docs = doc_processor.load_and_partition_documents(
                docs,
                partition_by_api=partition_by_api, 
                upload_bucket=bucket_name
            )
            chunk_obj, output_paths = doc_processor.chunk_documents(partitioned_docs)
                    
            # Index documents
            db_service.index_data(chunk_obj,batch_size=batch_size)
            
            end_time = time.time()
            elapsed_time = end_time - start_time
            st.markdown(f":heavy_check_mark: Loaded docs in {elapsed_time:.2f} seconds")
            
        except Exception as e:
            st.error(f"Error processing documents: {str(e)}")
            raise e
