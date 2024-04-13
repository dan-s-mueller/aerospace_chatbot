import os, sys, time
import glob
import streamlit as st
import streamlit_authenticator as stauth
import yaml
from yaml.loader import SafeLoader

from langchain_openai import OpenAIEmbeddings
from langchain_voyageai import VoyageAIEmbeddings
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings

sys.path.append('../../src/aerospace_chatbot')  # Add package to path
import admin, data_processing

# Page setup
current_directory = os.path.dirname(os.path.abspath(__file__))
home_dir = os.path.abspath(os.path.join(current_directory, "../../"))
paths,sb,secrets=admin.st_setup_page('Aerospace Chatbot',
                                     home_dir,
                                     {'vector_database':True,
                                      'embeddings':True,
                                      'rag_type':True,
                                      'index_name':True,
                                      'secret_keys':True})

# Read the user credentials from the config file, and authenticate the user
with open(os.path.join(paths['config_folder_path'],'users.yml')) as file:
    config = yaml.load(file, Loader=SafeLoader)
authenticator = stauth.Authenticate(
    config['credentials'],
    config['cookie']['name'],
    config['cookie']['key'],
    config['cookie']['expiry_days'])
name, authentication_status, username = authenticator.login()

# Store the user's name and authentication status in the session state
if st.session_state["authentication_status"]:
    # User logged in
    st.warning(f'''Welcome *{st.session_state["name"]}*. 
               You are in administrator mode and have the ability to create/modify/delete vector databases.
               Please be careful with your actions as they will be permanent.''')

    # Add section for connection status and vector database cleanup
    st.subheader('Connection status and vector database cleanup')
    admin.st_connection_status_expander(expanded=False,delete_buttons=True)

    # Add section for creating and loading into a vector database
    st.subheader('Create and load into a vector database')
    
    # Populate the main screen
    if sb["index_type"]=='RAGatouille':
        query_model=sb['query_model']
    elif sb['query_model']=='OpenAI' or sb['query_model']=='Voyage':
        if sb['query_model']=='OpenAI':
            query_model=OpenAIEmbeddings(model=sb['embedding_name'],
                                         openai_api_key=secrets['OPENAI_API_KEY'])
        elif sb['query_model']=='Voyage':
            # For voyage embedding truncation see here: https://docs.voyageai.com/docs/embeddings#python-api.
            # Leaving out trunction gives an error.
            query_model=VoyageAIEmbeddings(model=sb['embedding_name'],
                                           voyage_api_key=secrets['VOYAGE_API_KEY'],
                                           truncation=False)
        elif sb['query_model']=='Hugging Face':
            query_model = HuggingFaceInferenceAPIEmbeddings(model_name=sb['embedding_name'],
                                                            api_key=secrets['HUGGINGFACEHUB_API_TOKEN'])

    # Find docs
    data_folder = st.text_input('Enter a directory relative to the base directory',
                                os.path.join(paths['data_folder_path'],'AMS'),
                                help='Enter a directory, must be an absolute path.')
    if not os.path.isdir(data_folder):
        st.error('The entered directory does not exist')
    docs = glob.glob(os.path.join(data_folder,'*.pdf'))   # Only get the PDFs in the directory
    st.markdown('PDFs found: '+str(docs))
    st.markdown('Number of PDFs found: ' + str(len(docs)))
    database_appendix=st.text_input('Appendix for database name','ams')

    # Add an expandable box for options
    with st.expander("Options",expanded=True):
        clear_database = st.checkbox('Delete existing database?',value=True)
        batch_size=st.number_input('Batch size for upsert', 
                        min_value=1, step=1, value=50,
                        help='''The number of documents to upsert at a time. 
                                Useful for hosted databases (e.g. Pinecone), or those that require long processing times.''')
        
        # Merge pages before processing
        merge_pages=st.checkbox('Merge pages before processing?',value=False,
                                help='If checked, pages will be merged before processing.')
        if merge_pages:
            n_merge_pages=st.number_input('Number of pages to merge', min_value=2, step=1, value=2, 
                                            help='''Number of pages to merge into a single document. 
                                            This is done before chunking occurs. 
                                            If zero, each page is processed independently before chunking.''')
        else:
            n_merge_pages=None
        
        # For each rag_type, set chunk parameters
        if sb['rag_type']=='Standard':
            chunk_method= st.selectbox('Chunk method', ['character_recursive','None'], 
                                       index=0,
                                       help='''https://python.langchain.com/docs/modules/data_connection/document_transformers/. 
                                               None will take whole PDF pages as documents in the database.''')
            if chunk_method=='character_recursive':
                chunk_size=st.number_input('Chunk size (characters)', min_value=1, step=1, value=400, help='An average paragraph is around 400 characters.')
                chunk_overlap=st.number_input('Chunk overlap (characters)', min_value=0, step=1, value=0)
            elif chunk_method=='None':
                chunk_size=None
                chunk_overlap=None
            else:
                raise NotImplementedError
        elif sb['rag_type']=='Parent-Child':
            chunk_method= st.selectbox('Chunk method', ['character_recursive'], index=0,help='https://python.langchain.com/docs/modules/data_connection/document_transformers/')
            if chunk_method=='character_recursive':
                chunk_size=st.number_input('Chunk size (characters)', min_value=1, step=1, value=400, help='An average paragraph is around 400 characters.')
                chunk_overlap=st.number_input('Chunk overlap (characters)', min_value=0, step=1, value=0)
            else:
                raise NotImplementedError
        elif sb['rag_type']=='Summary':
            chunk_size=None
            chunk_overlap=None
        else:  
            raise NotImplementedError
        export_json = st.checkbox('Export jsonl?', value=True,help='If checked, a jsonl file will be generated when you load docs to vector database. No embeddeng data will be saved.')
        if export_json:
            json_file=st.text_input('Jsonl file',os.path.join(data_folder,f'{database_appendix}_data-{chunk_size}-{chunk_overlap}.jsonl'))
            json_file=os.path.join(paths['base_folder_path'],json_file)


    # Add a button to run the function
    if st.button('Load docs into vector database'):
        start_time = time.time()  # Start the timer

        if sb['rag_type']=='Summary':
            llm=admin.set_llm(sb,secrets,type='rag')
        else:
            llm=None

        data_processing.load_docs(sb['index_type'],
                            docs,
                            rag_type=sb['rag_type'],
                            query_model=query_model,
                            embedding_name=sb['embedding_name'],
                            index_name=sb['index_name']+'-'+database_appendix,
                            n_merge_pages=n_merge_pages,
                            chunk_method=chunk_method,
                            chunk_size=chunk_size,
                            chunk_overlap=chunk_overlap,                  
                            file_out=json_file,
                            clear=clear_database,
                            batch_size=batch_size,
                            local_db_path=paths['db_folder_path'],
                            llm=llm,
                            show_progress=True)
        end_time = time.time()  # Stop the timer
        elapsed_time = end_time - start_time 
        st.markdown(f":heavy_check_mark: Loaded docs in {elapsed_time:.2f} seconds")
elif st.session_state["authentication_status"] is False:
    st.error('Username/password is incorrect')
elif st.session_state["authentication_status"] is None:
    st.warning('Please enter your username and password')