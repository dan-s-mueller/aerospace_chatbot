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
    admin.st_connection_status_expander(expanded=True,delete_buttons=True)

    # Add section for creating and loading into a vector database
    st.subheader('Create and load into a vector database')

    # Set query model
    query_model = admin.get_query_model(sb, secrets)

    # Find docs
    data_folder = st.text_input('Enter a directory relative to the base directory',
                                os.path.join(paths['data_folder_path'],'AMS'),
                                help='Enter a directory, must be an absolute path.')
    if not os.path.isdir(data_folder):
        st.error('The entered directory does not exist')
    docs = glob.glob(os.path.join(data_folder,'*.pdf'))   # Only get the PDFs in the directory
    # TODO update so that you can select the files to upload
    st.markdown('PDFs found: '+str(docs))
    st.markdown('Number of PDFs found: ' + str(len(docs)))

    # Set database name
    index_appendix=st.text_input('Appendix for index name','ams')
    index_name = (sb['embedding_name'].replace('/', '-') + '-' + index_appendix).lower()

    # Add an expandable box for options
    with st.expander("Options",expanded=True):
        clear_database = st.checkbox('Delete existing database?',value=True)

        if sb['query_model']=='Hugging Face':
            batch_size_max=32
            batch_size=32
        else:
            batch_size_max=None
            batch_size=100
        batch_size=st.number_input('Batch size for upsert', 
                        min_value=1, max_value=batch_size_max, step=1, value=batch_size,
                        help='''The number of documents to upsert at a time. 
                                Useful for hosted databases (e.g. Pinecone), or those that require long processing times.
                                When using hugging face embeddings, batch size maximum is 32.''')
        
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
            chunk_method=None
            chunk_size=None
            chunk_overlap=None
            sb['model_options']={}
            sb['model_options']['temperature'] = st.slider('Summary model remperature', min_value=0.0, max_value=2.0, value=0.1, step=0.1,help='Temperature for LLM.')
            sb['model_options']['output_level'] = st.number_input('Summary model max output tokens', min_value=50, step=10, value=4000,
                                                help='Max output tokens for LLM. Concise: 50, Verbose: >1000. Limit depends on model.')
        else:  
            raise NotImplementedError
        export_json = st.checkbox('Export jsonl?', value=True,help='If checked, a jsonl file will be generated when you load docs to vector database. No embeddeng data will be saved.')
        if export_json:
            json_file=st.text_input('Jsonl file',os.path.join(data_folder,f'{index_appendix}_data-{chunk_size}-{chunk_overlap}.jsonl'))
            json_file=os.path.join(paths['base_folder_path'],json_file)

        # Set LLM if relevant
        if sb['rag_type']=='Summary':
            llm=admin.set_llm(sb,secrets,type='rag')
            print(llm)
        else:
            llm=None

    # Check the index name, give error before running if it is invalid
    try:
        if sb['rag_type'] == 'Parent-Child':
            index_name_check = index_name + '-parent-child'
        if sb['rag_type'] == 'Summary':
            index_name_check = index_name + '-' + llm.model_name.replace('.', '-').replace('/', '-')[:6].lower() + '-summary'
        else:
            index_name_check = index_name
        index_name_disp=data_processing.check_db_name(sb['index_type'],index_name_check)
        st.markdown(f'Index name: {index_name_disp}')
    except ValueError as e:
        st.warning(str(e))
        st.stop()

        

    # Add a button to run the function
    if st.button('Load docs into vector database'):
        start_time = time.time()  # Start the timer

        data_processing.load_docs(sb['index_type'],
                            docs,
                            query_model,
                            rag_type=sb['rag_type'],
                            index_name=index_name,
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