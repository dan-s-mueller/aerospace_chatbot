import os, sys
import streamlit as st

from langchain_openai import OpenAIEmbeddings
from langchain_voyageai import VoyageAIEmbeddings

sys.path.append('../src/aerospace_chatbot')   # Add package to path
import admin, data_processing

# Page setup
st.title('📈 Visualize Data')
current_directory = os.path.dirname(os.path.abspath(__file__))
home_dir = os.path.abspath(os.path.join(current_directory, "../../"))

# Initialize SidebarManager
sidebar_manager = admin.SidebarManager(st.session_state.config_file)

# Get paths, sidebar values, and secrets
paths = sidebar_manager.get_paths(home_dir)
sb = sidebar_manager.render_sidebar()
secrets = sidebar_manager.get_secrets()

# Set up session state variables
if 'viewer' not in st.session_state:
    st.session_state.viewer = None

# Set the query model
if sb["index_type"]=='RAGatouille':
    raise Exception('RAGatouille not supported for this function.')
# elif sb["index_type"]=='Pinecone':
#     raise Exception('Only index type ChromaDB is supported for this function.')
elif sb['query_model']=='OpenAI' or sb['query_model']=='Voyage':
    if sb['query_model']=='OpenAI':
        query_model=OpenAIEmbeddings(model=sb['embedding_name'],openai_api_key=secrets['OPENAI_API_KEY'])
    elif sb['query_model']=='Voyage':
        query_model=VoyageAIEmbeddings(model='voyage-2', voyage_api_key=secrets['VOYAGE_API_KEY'])
else:
    raise Exception('Unsupported query model for visualization.')

# st.info('Visualization is only functional with ChromaDB index type.')

llm=admin.set_llm(sb,secrets)    # Set the LLM
query_model = admin.get_query_model(sb, secrets)    # Set query model

# Add options
export_file=st.checkbox('Export local dataset file?',value=False,help='Export the data, including embeddings to a parquet file')
if export_file:
    file_name=st.text_input('Enter the file name',value=f"{os.path.join(paths['data_folder_path'],sb['index_selected']+'.parquet')}")
hf_dataset=st.checkbox('Export dataset to Hugging Face?',value=False,help='Export the data, including embeddings to a Hugging Face dataset.')

if hf_dataset:
    hf_org_name=st.text_input('Enter the Hugging Face organization name',value='ai-aerospace',help='The organization name on Hugging Face.')
    dataset_name=st.text_input('Enter the dataset name',value=sb['index_selected'],help='The name of the dataset to be created on Hugging Face. Output will be sidebar selection. Will be appended with ac-.')
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
    with st.status('Processing visualization...',expanded=True):
        st.markdown('Generating visualization data...')
        df = data_processing.get_docs_questions_df(
            sb['index_type'],
            paths['db_folder_path'],
            sb['index_selected'],
            paths['db_folder_path'],
            sb['index_selected']+'-queries',
            query_model
        )
        if cluster_data:
            st.markdown('Clustering data...')
            df=data_processing.add_clusters(df,n_clusters,
                                            label_llm=llm,
                                            doc_per_cluster=docs_per_cluster)
        
        if export_file:
            st.markdown('Clustering data...')
            df.to_parquet(file_name)

        if hf_dataset:
            st.markdown('Uploading Hugging Face dataset...')
            data_processing.export_to_hf_dataset(df,dataset_name)
            st.markdown(f"The dataset is uploaded to: {'https://huggingface.co/datasets/'+dataset_name}")

    if spotlight_viewer or st.session_state.viewer is not None: 
        st.markdown(f"Spotlight running on: {'http://'+'localhost'+':'+str(port)}")
        st.info('Functionality only works with locally deployed versions.')
        st.session_state.viewer = data_processing.get_or_create_spotlight_viewer(df,port=port)
    else:
        st.info('Functionality only works with locally deployed versions.')