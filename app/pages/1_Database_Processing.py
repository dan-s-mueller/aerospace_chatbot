import os, sys, time, logging
import glob
import streamlit as st
import streamlit_authenticator as stauth
import yaml
from yaml.loader import SafeLoader
from dotenv import load_dotenv,find_dotenv

from langchain_openai import OpenAIEmbeddings
from langchain_voyageai import VoyageAIEmbeddings

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
    admin.st_connection_status_expander(paths['data_folder_path'],delete_buttons=True)

    # Add section for creating and loading into a vector database
    st.subheader('Create and load into a vector database')
    
    # Populate the main screen
    logging.info(f'index_type test, {sb["index_type"]}')

    if sb["index_type"]=='RAGatouille':
        logging.info('Set hugging face model for queries.')
        query_model=sb['query_model']
    elif sb['query_model']=='Openai' or 'Voyage':
        logging.info('Set embeddings model for queries.')
        if sb['query_model']=='Openai':
            query_model=OpenAIEmbeddings(model=sb['embedding_name'],openai_api_key=secrets['OPENAI_API_KEY'])
        elif sb['query_model']=='Voyage':
            query_model=VoyageAIEmbeddings(model='voyage-2',voyage_api_key=secrets['VOYAGE_API_KEY'])
    logging.info('Query model set: '+str(query_model))

    # Find docs
    data_folder = st.text_input('Enter a directory relative to the base directory',
                                os.path.join(paths['data_folder_path'],'AMS'),
                                help='Enter a directory, must be an absolute path.')
    if not os.path.isdir(data_folder):
        st.error('The entered directory does not exist')
    docs = glob.glob(os.path.join(data_folder,'*.pdf'))   # Only get the PDFs in the directory
    st.markdown('PDFs found: '+str(docs))
    st.markdown('Number of PDFs found: ' + str(len(docs)))
    logging.info('Docs: '+str(docs))
    database_appendix=st.text_input('Appendix for database name','ams')

    # Add an expandable box for options
    with st.expander("Options",expanded=True):
        clear_database = st.checkbox('Delete existing database?',value=True)
        if sb['query_model']=='Openai' or 'ChromaDB':
            # OpenAI will time out if the batch size is too large
            batch_size=st.number_input('Batch size for upsert', min_value=1, step=1, value=50)
        else:
            batch_size=None
        
        if sb['rag_type']!='Summary':
            chunk_method= st.selectbox('Chunk method', ['character_recursive'], index=0,help='https://python.langchain.com/docs/modules/data_connection/document_transformers/')
            if chunk_method=='character_recursive':
                chunk_size=st.number_input('Chunk size (characters)', min_value=1, step=1, value=400, help='An average paragraph is around 400 characters.')
                chunk_overlap=st.number_input('Chunk overlap (characters)', min_value=0, step=1, value=0)
            else:
                raise NotImplementedError
        else:
            chunk_size=None
            chunk_overlap=None
        export_json = st.checkbox('Export jsonl?', value=True,help='If checked, a jsonl file will be generated when you load docs to vector database.')
        if export_json:
            json_file=st.text_input('Jsonl file',os.path.join(data_folder,'ams_data-400-0.jsonl'))
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
                            index_name=sb['index_name']+'-'+database_appendix,
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