import os
import streamlit as st

from aerospace_chatbot.ui import SidebarManager, get_or_create_spotlight_viewer, handle_sidebar_state
from aerospace_chatbot.services import EmbeddingService, LLMService, DatabaseService

# Page setup
st.title('📈 Visualize Data')
current_directory = os.path.dirname(os.path.abspath(__file__))
home_dir = os.path.abspath(os.path.join(current_directory, "../../"))

# Initialize SidebarManager first
if 'sidebar_state_initialized' not in st.session_state:
    st.session_state.sidebar_manager = SidebarManager(st.session_state.config_file)
    st.session_state.sb = {}

# Handle sidebar state
st.session_state.sb = handle_sidebar_state(st.session_state.sidebar_manager)

# Set up session state variables
if 'viewer' not in st.session_state:
    st.session_state.viewer = None

# Initialize services after potential rerun
if st.session_state.sb["index_type"] == 'RAGatouille':
    raise Exception('RAGatouille not supported for this function.')

embedding_service = EmbeddingService(
    model_service=st.session_state.sb['embedding_service'],
    model=st.session_state.sb['embedding_model']
)

llm_service = LLMService(
    model_service=st.session_state.sb['llm_service'],
    model=st.session_state.sb['llm_model'],
    temperature=st.session_state.sb['model_options']['temperature'],
    max_tokens=st.session_state.sb['model_options']['output_level']
)

db_service = DatabaseService(
    db_type=st.session_state.sb['index_type'],
    index_name=st.session_state.sb['index_selected'],
    rag_type=st.session_state.sb['rag_type'],
    embedding_service=embedding_service,
    doc_type='document'
)

# Add options
export_file=st.checkbox('Export local dataset file?',value=False,help='Export the data, including embeddings to a parquet file')
if export_file:
    file_name=st.text_input('Enter the file name',value=f"{os.path.join(os.getenv('LOCAL_DB_PATH'),st.session_state.sb['index_selected']+'.parquet')}")
hf_dataset=st.checkbox('Export dataset to Hugging Face?',value=False,help='Export the data, including embeddings to a Hugging Face dataset.')

if hf_dataset:
    hf_org_name=st.text_input('Enter the Hugging Face organization name',value='ai-aerospace',help='The organization name on Hugging Face.')
    dataset_name=st.text_input('Enter the dataset name',value=st.session_state.sb['index_selected'],help='The name of the dataset to be created on Hugging Face. Output will be sidebar selection. Will be appended with ac-.')
    dataset_name=hf_org_name+'/'+'ac-'+dataset_name

cluster_data=st.checkbox('Cluster data?',value=False,help='Cluster the data using the embeddings using KMeans clustering.')
if cluster_data:
    st.markdown('LLM to be used for clustering is set in sidebar.')
    n_clusters=st.number_input('Enter the number of clusters',value=10)
    docs_per_cluster=st.number_input('Enter the number of documents per cluster to generate label',value=10)

spotlight_viewer=st.checkbox('Launch Spotlight viewer?',value=False,help='Launch the Spotlight viewer to visualize the data. If unselected, will provide link to static Spotlight viewer.')
if spotlight_viewer:
    port=st.number_input('Enter the port number',value=9000,help='The port number to run the viewer on. Default to 9000.')

if st.button('Visualize'):
    with st.status('Processing visualization...', expanded=True):
        st.markdown('Generating visualization data...')
        
        # Get documents and questions from database
        df = db_service.get_docs_questions_df(
            index_name=st.session_state.sb['index_selected'],
            query_index_name=st.session_state.sb['index_selected']+'-queries',
            embedding_service=embedding_service
        )
        
        if cluster_data:
            st.markdown('Clustering data...')
            df = db_service.add_clusters(
                df,
                n_clusters=n_clusters,
                llm_service=llm_service,
                docs_per_cluster=docs_per_cluster
            )
        
        if export_file:
            st.markdown('Exporting to file...')
            df.to_parquet(file_name)

        if hf_dataset:
            st.markdown('Uploading Hugging Face dataset...')
            db_service.export_to_hf_dataset(df, dataset_name)
            st.markdown(f"The dataset is uploaded to: {'https://huggingface.co/datasets/'+dataset_name}")

    if spotlight_viewer or st.session_state.viewer is not None:
        st.markdown(f"Spotlight running on: {'http://'+'localhost'+':'+str(port)}")
        st.info('Functionality only works with locally deployed versions.')
        st.session_state.viewer = get_or_create_spotlight_viewer(df, port=port)
    else:
        st.info('Functionality only works with locally deployed versions.')