import data_processing
from prompts import TEST_QUERY_PROMPT

import inspect
import os
import json
import streamlit as st
from dotenv import load_dotenv,find_dotenv

import openai
from pinecone import Pinecone
import chromadb
from langchain_openai import ChatOpenAI

from langchain_openai import OpenAIEmbeddings
from langchain_voyageai import VoyageAIEmbeddings
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from ragatouille import RAGPretrainedModel

class SecretKeyException(Exception):
    """Exception raised for secret key related errors.

    Attributes:
        message -- explanation of the error
        id -- unique identifier for the error
    """

    def __init__(self, message, id):
        super().__init__(message)
        self.id = id
class DatabaseException(Exception):
    """Exception raised for database related errors.

    Attributes:
        message -- explanation of the error
        id -- unique identifier for the error
    """

    def __init__(self, message, id):
        super().__init__(message)
        self.id = id
def load_sidebar(config_file,
                 vector_database=False,
                 embeddings=False,
                 rag_type=False,
                 index_selected=False,
                 llm=False,
                 model_options=False,
                 secret_keys=False):
    """
    Loads the sidebar configuration for the chatbot.

    Args:
        config_file (str): The path to the configuration file.
        vector_database (bool, optional): Whether to include the vector database in the sidebar. Defaults to False.
        embeddings (bool, optional): Whether to include embeddings in the sidebar. Defaults to False.
        rag_type (bool, optional): Whether to include RAG type in the sidebar. Defaults to False.
        index_selected (bool, optional): Whether to include index name in the sidebar. Defaults to False.
        llm (bool, optional): Whether to include LLM in the sidebar. Defaults to False.
        model_options (bool, optional): Whether to include model options in the sidebar. Defaults to False.
        secret_keys (bool, optional): Whether to include secret keys in the sidebar. Defaults to False.

    Returns:
        dict: The sidebar configuration.
    """
    sb_out={}
    with open(config_file, 'r') as f:
        config = json.load(f)
        databases = {db['name']: db for db in config['databases']}
        embeddings_list = {e['name']: e for e in config['embeddings']}
        llms  = {m['name']: m for m in config['llms']}

    # Set local db path
    if os.getenv('LOCAL_DB_PATH') is None or os.getenv('LOCAL_DB_PATH')=='':
        # This is the case where the .env file is not in the directory
        raise SecretKeyException('Local Database Path is required. Use an absolute path for local, or /data for hugging face spaces.','LOCAL_DB_PATH_MISSING')

    # Vector databases
    if vector_database:
        st.sidebar.title('Vector database')
        sb_out['index_type']=st.sidebar.selectbox('Index type', list(databases.keys()), index=1,help='Select the type of index to use.')

        if embeddings:
            # Embeddings
            st.sidebar.title('Embeddings',help='See embedding leaderboard here for performance overview: https://huggingface.co/spaces/mteb/leaderboard')
            if sb_out['index_type']=='RAGatouille':    # Default to selecting hugging face model for RAGatouille, otherwise select alternates
                sb_out['query_model']=st.sidebar.selectbox('Hugging face rag models', 
                                                        databases[sb_out['index_type']]['embedding_models'], 
                                                        index=0,
                                                        help="Models listed are compatible with the selected index type.")
                sb_out['embedding_name']=sb_out['query_model']
            else:
                sb_out['query_model']=st.sidebar.selectbox('Embedding model family', 
                                                        databases[sb_out['index_type']]['embedding_models'], 
                                                        index=0,
                                                        help="Model provider.")
                sb_out['embedding_name']=st.sidebar.selectbox('Embedding model', 
                                                        embeddings_list[sb_out['query_model']]['embedding_models'], 
                                                        index=0,
                                                        help="Models listed are compatible with the selected index type.")
            
        if rag_type:
            # RAG Type
            st.sidebar.title('RAG Type')
            if sb_out['index_type']=='RAGatouille':
                sb_out['rag_type']=st.sidebar.selectbox('RAG type', ['Standard'], index=0,help='Only Standard is available for RAGatouille.')
            else:
                sb_out['rag_type']=st.sidebar.selectbox('RAG type', ['Standard','Parent-Child','Summary'], index=0,help='Parent-Child is for parent-child RAG. Summary is for summarization RAG.')
                if sb_out['rag_type']=='Summary' or sb_out['rag_type']=='Muti-Query':
                    sb_out['rag_llm_source']=st.sidebar.selectbox('RAG LLM model', list(llms.keys()), index=0,help='Select the LLM model for RAG.')
                    if sb_out['rag_llm_source']=='OpenAI':
                        sb_out['rag_llm_model']=st.sidebar.selectbox('RAG OpenAI model', llms[sb_out['rag_llm_source']]['models'], index=0,help='Select the OpenAI model for RAG.')
                    if sb_out['rag_llm_source']=='Hugging Face':
                        sb_out['rag_llm_model']=st.sidebar.selectbox('RAG Hugging Face model', 
                                                                llms['Hugging Face']['models'], 
                                                                index=0,
                                                                help='Select the Hugging Face model for RAG.')
                        sb_out['rag_hf_endpoint']='https://api-inference.huggingface.co/v1'
                    elif sb_out['rag_llm_source']=='LM Studio (local)':
                        sb_out['rag_llm_model']=st.sidebar.text_input('Local host URL',
                                                                'http://localhost:1234/v1',
                                                                help='See LM studio configuration for local host URL.')
                        st.sidebar.warning('You must load a model in LM studio first for this to work.')
        if index_selected:
            if embeddings and rag_type:
                # Index Name 
                st.sidebar.title('Index Selected')  
                name=[]
                # For each index type, list indices available for the base name
                if sb_out['index_type']=='ChromaDB':
                    indices=show_chroma_collections(format=False)
                    if indices['status']:
                        for index in indices['message']:
                            # Be compatible with embedding types already used. Pinecone only supports lowercase.
                            if index.name.startswith((sb_out['embedding_name'].replace('/', '-')).lower()):    
                                if not index.name.endswith('-queries'): # Don't list query database as selectable
                                    if sb_out['rag_type']=='Parent-Child':
                                        if index.name.endswith('-parent-child'):
                                            name.append(index.name)
                                    elif sb_out['rag_type']=='Summary':
                                        if index.name.endswith('-summary'):
                                            name.append(index.name)
                                    else:
                                        name.append(index.name)
                        sb_out['index_selected']=st.sidebar.selectbox('Index selected',name,index=0,help='Select the index to use for the application.')
                    else:
                        st.sidebar.markdown('No collections found.',help='Check the status on Home.')
                elif sb_out['index_type']=='Pinecone':
                    indices=show_pinecone_indexes(format=False)
                    if indices['status']:
                        for index in indices['message']:
                            # if index['status']['state']=='Ready':
                            #     name.append(index['name'])
                            if index['name'].startswith((sb_out['embedding_name'].replace('/', '-')).lower()):    
                                if not index['name'].endswith('-queries'): # Don't list query database as selectable
                                    if sb_out['rag_type']=='Parent-Child':
                                        if index['name'].endswith('-parent-child'):
                                            name.append(index['name'])
                                    elif sb_out['rag_type']=='Summary':
                                        if index['name'].endswith('-summary'):
                                            name.append(index['name'])
                                    else:
                                        name.append(index['name'])
                        sb_out['index_selected']=st.sidebar.selectbox('Index selected',name,index=0,help='Select the index to use for the application.')
                elif sb_out['index_type']=='RAGatouille':
                    indices=show_ragatouille_indexes(format=False)
                    if len(indices)>0:
                        for index in indices['message']:
                            # Be compatible with embedding types already used. Pinecone only supports lowercase.
                            if index.startswith((sb_out['embedding_name'].replace('/', '-')).lower()):    
                                name.append(index)
                        sb_out['index_selected']=st.sidebar.selectbox('Index selected',name,index=0,help='Select the index to use for the application.')
                    else:
                        st.sidebar.markdown('No collections found.',help='Check the status on Home.')
                else:
                    raise NotImplementedError
                try:
                    if not name:
                        raise DatabaseException('No collections found for the selected index type/embedding. Create a new database, or select another index type/embedding.','NO_COMPATIBLE_COLLECTIONS')
                except DatabaseException as e:
                    st.warning(f"{e}")
                    st.stop()
            else:
                raise ValueError('Embeddings must be enabled to select an index name.')
        if llm:
            # LLM
            st.sidebar.title('LLM',help='See LLM leaderboard here for performance overview: https://huggingface.co/spaces/lmsys/chatbot-arena-leaderboard')
            sb_out['llm_source']=st.sidebar.selectbox('LLM model', list(llms.keys()), index=0,help='Select the LLM model for the application.')
            if sb_out['llm_source']=='OpenAI':
                sb_out['llm_model']=st.sidebar.selectbox('OpenAI model', llms[sb_out['llm_source']]['models'], index=0,help='Select the OpenAI model for the application.')
            elif sb_out['llm_source']=='Hugging Face':
                sb_out['llm_model']=st.sidebar.selectbox('Hugging Face model', 
                                                        llms['Hugging Face']['models'], 
                                                        index=0,
                                                        help='Select the Hugging Face model for the application.')
                sb_out['hf_endpoint']='https://api-inference.huggingface.co/v1'
            elif sb_out['llm_source']=='LM Studio (local)':
                sb_out['llm_model']=st.sidebar.text_input('Local host URL',
                                                        'http://localhost:1234/v1',
                                                        help='See LM studio configuration for local host URL.')
                st.sidebar.warning('You must load a model in LM studio first for this to work.')
        if model_options:
            # Add input fields in the sidebar
            st.sidebar.title('LLM Options')
            temperature = st.sidebar.slider('Temperature', min_value=0.0, max_value=2.0, value=0.1, step=0.1,help='Temperature for LLM.')
            output_level = st.sidebar.number_input('Max output tokens', min_value=50, step=10, value=1000,
                                                help='Max output tokens for LLM. Concise: 50, Verbose: 1000. Limit depends on model.')
            # Set different options for if ragatouille is used, since it has fewer parameters to select
            if 'index_type' in sb_out:
                st.sidebar.title('Retrieval Options')
                k = st.sidebar.number_input('Number of items per prompt', min_value=1, step=1, value=4,help='Number of items to retrieve per query.')
                if sb_out['index_type']!='RAGatouille':
                    search_type = st.sidebar.selectbox('Search Type', ['similarity', 'mmr'], index=0,help='Select the search type for the application.')
                    sb_out['model_options']={'output_level':output_level,
                                            'k':k,
                                            'search_type':search_type,
                                            'temperature':temperature}
                else:
                    sb_out['model_options']={'output_level':output_level,
                                             'k':k,
                                            'search_type':None,
                                            'temperature':temperature}
    else:
        if embeddings or rag_type or index_selected or llm or model_options:
            # Must have vector database for any of this functionality.
            raise ValueError('Vector database must be enabled to use these options.')

    # Secret keys, which does not rely on vector_database
    if secret_keys:
        sb_out['keys']={}
        # Add a section for secret keys
        st.sidebar.title('Secret keys',help='See Home page under Connection Status for status of keys.')
        st.sidebar.markdown('If .env file is in directory, will use that first.')
        if 'llm_source' in sb_out and sb_out['llm_source'] == 'OpenAI':
            sb_out['keys']['OPENAI_API_KEY'] = st.sidebar.text_input('OpenAI API Key', type='password',help='OpenAI API Key: https://platform.openai.com/api-keys')
        elif 'query_model' in sb_out and sb_out['query_model'] == 'OpenAI':
            sb_out['keys']['OPENAI_API_KEY'] = st.sidebar.text_input('OpenAI API Key', type='password',help='OpenAI API Key: https://platform.openai.com/api-keys')
        if 'llm_source' in sb_out and sb_out['llm_source']=='Hugging Face':
            sb_out['keys']['HUGGINGFACEHUB_API_TOKEN'] = st.sidebar.text_input('Hugging Face API Key', type='password',help='Hugging Face API Key: https://huggingface.co/settings/tokens')
        if 'query_model' in sb_out and sb_out['query_model']=='Voyage':
            sb_out['keys']['VOYAGE_API_KEY'] = st.sidebar.text_input('Voyage API Key', type='password',help='Voyage API Key: https://dash.voyageai.com/api-keys')
        if 'index_type' in sb_out and sb_out['index_type']=='Pinecone':
            sb_out['keys']['PINECONE_API_KEY']=st.sidebar.text_input('Pinecone API Key',type='password',help='Pinecone API Key: https://www.pinecone.io/')
            
    return sb_out
def set_secrets(sb):
    """
    Sets the secrets for various API keys by retrieving them from the environment variables or the sidebar.

    Args:
        sb (dict): The sidebar data containing the API keys.

    Returns:
        dict: A dictionary containing the set API keys.

    Raises:
        SecretKeyException: If any of the required API keys are missing.
    """
    secrets={}

    secrets['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')
    if not secrets['OPENAI_API_KEY'] and 'keys' in sb and 'OPENAI_API_KEY' in sb['keys']:
        secrets['OPENAI_API_KEY'] = sb['keys']['OPENAI_API_KEY']
        os.environ['OPENAI_API_KEY'] = secrets['OPENAI_API_KEY']
        if os.environ['OPENAI_API_KEY']=='':
            raise SecretKeyException('OpenAI API Key is required.','OPENAI_API_KEY_MISSING')
    openai.api_key = secrets['OPENAI_API_KEY']

    secrets['VOYAGE_API_KEY'] = os.getenv('VOYAGE_API_KEY')
    if not secrets['VOYAGE_API_KEY'] and 'keys' in sb and 'VOYAGE_API_KEY' in sb['keys']:
        secrets['VOYAGE_API_KEY'] = sb['keys']['VOYAGE_API_KEY']
        os.environ['VOYAGE_API_KEY'] = secrets['VOYAGE_API_KEY']
        if os.environ['VOYAGE_API_KEY']=='':
            raise SecretKeyException('Voyage API Key is required.','VOYAGE_API_KEY_MISSING')

    secrets['PINECONE_API_KEY'] = os.getenv('PINECONE_API_KEY')
    if not secrets['PINECONE_API_KEY'] and 'keys' in sb and 'PINECONE_API_KEY' in sb['keys']:
        secrets['PINECONE_API_KEY'] = sb['keys']['PINECONE_API_KEY']
        os.environ['PINECONE_API_KEY'] = secrets['PINECONE_API_KEY']
        if os.environ['PINECONE_API_KEY']=='':
            raise SecretKeyException('Pinecone API Key is required.','PINECONE_API_KEY_MISSING')

    secrets['HUGGINGFACEHUB_API_TOKEN'] = os.getenv('HUGGINGFACEHUB_API_TOKEN')
    if not secrets['HUGGINGFACEHUB_API_TOKEN'] and 'keys' in sb and 'HUGGINGFACEHUB_API_TOKEN' in sb['keys']:
        secrets['HUGGINGFACEHUB_API_TOKEN'] = sb['keys']['HUGGINGFACEHUB_API_TOKEN']
        os.environ['HUGGINGFACEHUB_API_TOKEN'] = secrets['HUGGINGFACEHUB_API_TOKEN']
        if os.environ['HUGGINGFACEHUB_API_TOKEN']=='':
            raise SecretKeyException('Hugging Face API Key is required.','HUGGINGFACE_API_KEY_MISSING')
    return secrets
def test_key_status():
    """
    Check the status of various API keys based on environment variables.

    Returns:
        dict: A dictionary containing the status of each API key.
    """
    key_status = {}
    # OpenAI
    if os.getenv('OPENAI_API_KEY') is None or os.getenv('OPENAI_API_KEY') == '':
        key_status['OpenAI API Key'] = {'status': False}
    else:
        key_status['OpenAI API Key'] = {'status': True}

    # Voyage
    if os.getenv('VOYAGE_API_KEY') is None or os.getenv('VOYAGE_API_KEY') == '':
        key_status['Voyage API Key'] = {'status': False}
    else:
        key_status['Voyage API Key'] = {'status': True}

    # Pinecone
    if os.getenv('PINECONE_API_KEY') is None or os.getenv('PINECONE_API_KEY') == '':
        key_status['Pinecone API Key'] = {'status': False}
    else:
        key_status['PINECONE_API_KEY'] = {'status': True}
    
    # Hugging Face
    if os.getenv('HUGGINGFACEHUB_API_TOKEN') is None or os.getenv('HUGGINGFACEHUB_API_TOKEN') == '':
        key_status['Hugging Face API Key'] = {'status': False}
    else:
        key_status['Hugging Face API Key'] = {'status': True}

    return _format_key_status(key_status)
def set_llm(sb, secrets, type='prompt'):
    """
    Sets up and returns a language model (LLM) based on the provided parameters.

    Args:
        sb (dict): The configuration settings for the chatbot.
        secrets (dict): The secret keys and tokens required for authentication.
        type (str, optional): The type of LLM to set up. Defaults to 'prompt'.

    Returns:
        ChatOpenAI: The configured language model.

    Raises:
        ValueError: If an invalid LLM source is specified.

    """
    if type == 'prompt':  # use for prompting in chat applications
        if sb['llm_source'] == 'OpenAI':
            llm = ChatOpenAI(model_name=sb['llm_model'],
                             temperature=sb['model_options']['temperature'],
                             openai_api_key=secrets['OPENAI_API_KEY'],
                             max_tokens=sb['model_options']['output_level'],
                             tags=[sb['llm_model']])
        elif sb['llm_source'] == 'Hugging Face':
            llm = ChatOpenAI(base_url=sb['hf_endpoint'],
                             model=sb['llm_model'],
                             api_key=secrets['HUGGINGFACEHUB_API_TOKEN'],
                             temperature=sb['model_options']['temperature'],
                             max_tokens=sb['model_options']['output_level'],
                             tags=[sb['llm_model']])
        elif sb['llm_source'] == 'LM Studio (local)':
            # base_url takes local configuration from lm studio, no api key required.
            llm = ChatOpenAI(base_url=sb['llm_model'],
                             temperature=sb['model_options']['temperature'],
                             max_tokens=sb['model_options']['output_level'])
        else:
            raise ValueError("Invalid LLM source specified.")
    elif type == 'rag':   # use for RAG application (summary)
        if sb['rag_llm_source'] == 'OpenAI':
            llm = ChatOpenAI(model_name=sb['rag_llm_model'],
                             openai_api_key=secrets['OPENAI_API_KEY'],
                             tags=[sb['rag_llm_model']])
        elif sb['rag_llm_source'] == 'Hugging Face':
            llm = ChatOpenAI(base_url=sb['rag_hf_endpoint'],
                             model=sb['rag_llm_model'],
                             api_key=secrets['HUGGINGFACEHUB_API_TOKEN'],
                             tags=[sb['rag_llm_model']])
        elif sb['rag_llm_source'] == 'LM Studio (local)':
            # base_url takes local configuration from lm studio, no api key required.
            llm = ChatOpenAI(base_url=sb['rag_llm_model'],
                             model_name='lm_studio')
        else:
            raise ValueError("Invalid LLM source specified.")
    return llm
def get_query_model(sb, secrets):
    """
    Returns the query model based on the provided parameters.

    Args:
        sb (dict): A dictionary containing the parameters for the query model.
        secrets (dict): A dictionary containing the API keys for different query models.

    Returns:
        query_model: The selected query model based on the provided parameters.

    Raises:
        NotImplementedError: If the query model is not recognized.
    """
    if sb['index_type'] == 'RAGatouille':
        query_model = RAGPretrainedModel.from_pretrained(sb['embedding_name'],
                                                         index_root=os.path.join(os.getenv('LOCAL_DB_PATH'),'.ragatouille'))
    elif sb['query_model'] == 'OpenAI':
        query_model = OpenAIEmbeddings(model=sb['embedding_name'], openai_api_key=secrets['OPENAI_API_KEY'])
    elif sb['query_model'] == 'Voyage':
        query_model = VoyageAIEmbeddings(model=sb['embedding_name'], voyage_api_key=secrets['VOYAGE_API_KEY'], truncation=False)
    elif sb['query_model'] == 'Hugging Face':
        query_model = HuggingFaceInferenceAPIEmbeddings(model_name=sb['embedding_name'], api_key=secrets['HUGGINGFACEHUB_API_TOKEN'])
    else:
        raise NotImplementedError('Query model not recognized.')
    return query_model
def show_pinecone_indexes(format=True):
    """
    Retrieves the list of Pinecone indexes and their status.
    LOCAL_DB_PATH environment variable used to pass the local database path.

    Args:
        format (bool, optional): Specifies whether to format the output. Defaults to True.

    Returns:
        dict or str: If format is True, returns a formatted string representation of the Pinecone status.
                    If format is False, returns a dictionary containing the Pinecone status.

    """
    if os.getenv('PINECONE_API_KEY') is None or os.getenv('PINECONE_API_KEY')=='':
        pinecone_status = {'status': False, 'message': 'Pinecone API Key is not set.'}
    else:
        pc=Pinecone(api_key=os.getenv('PINECONE_API_KEY'))
        indexes=pc.list_indexes()
        if len(indexes)==0:
            pinecone_status = {'status': False, 'message': 'No indexes found'}
        else:
            pinecone_status = {'status': True, 'message': indexes}
    
    if format:
        return _format_pinecone_status(pinecone_status)
    else:
        return pinecone_status
def show_chroma_collections(format=True):
    """
    Retrieves the list of chroma collections from the local database.
    LOCAL_DB_PATH environment variable used to pass the local database path.

    Args:
        format (bool, optional): Specifies whether to format the output. Defaults to True.

    Returns:
        dict or str: If format is True, returns a formatted string representation of the chroma status.
                    If format is False, returns a dictionary containing the chroma status.

    Raises:
        ValueError: If the chroma vector database needs to be reset.

    """
    if os.getenv('LOCAL_DB_PATH') is None or os.getenv('LOCAL_DB_PATH')=='':
        chroma_status = {'status': False, 'message': 'Local database path is not set.'}
    else:
        db_folder_path=os.getenv('LOCAL_DB_PATH')
        try:
            persistent_client = chromadb.PersistentClient(path=os.path.join(db_folder_path,'chromadb'))
        except:
            raise ValueError("Chroma vector database needs to be reset, or the database path is incorrect. Clear cache, or reset path. You may have specified a path which is read only or has no collections.")
        collections=persistent_client.list_collections()
        if len(collections)==0:
            chroma_status = {'status': False, 'message': 'No collections found'}
        else:   
            chroma_status = {'status': True, 'message': collections}
    if format:
        return _format_chroma_status(chroma_status)
    else:
        return chroma_status
def show_ragatouille_indexes(format=True):
    """
    Retrieves the list of ragatouille indexes.
    LOCAL_DB_PATH environment variable used to pass the local database path.

    Args:
        format (bool, optional): Specifies whether to format the indexes. Defaults to True.

    Returns:
        dict or str: If format is True, returns a formatted string representation of the ragatouille status.
                    If format is False, returns a dictionary containing the ragatouille status.

    Raises:
        ValueError: If the ragatouille vector database needs to be reset.

    """
    if os.getenv('LOCAL_DB_PATH') is None or os.getenv('LOCAL_DB_PATH')=='':
        ragatouille_status = {'status': False, 'message': 'Local database path is not set.'}
    else:
        db_folder_path=os.getenv('LOCAL_DB_PATH')
        try:
            path=os.path.join(db_folder_path,'.ragatouille/colbert/indexes')
            indexes = []
            for item in os.listdir(path):
                item_path = os.path.join(path, item)
                if os.path.isdir(item_path):
                    indexes.append(item)
            if len(indexes)>0:
                ragatouille_status = {'status': True, 'message': indexes}
            else:
                ragatouille_status = {'status': False, 'message': 'No indexes found.'}  # if .ragatouille structure exists but is empty
        except:
            ragatouille_status = {'status': False, 'message': 'No indexes found.'}  # if .ragatouille structure does not exist yet

    if format:
        return _format_ragatouille_status(ragatouille_status)
    else:
        return ragatouille_status
def st_connection_status_expander(expanded: bool = True, delete_buttons: bool = False, set_secrets: bool = False):
    """
    Expands a Streamlit expander widget to display connection status information.
    LOCAL_DB_PATH environment variable used to pass the local database path.

    Args:
        expanded (bool, optional): Whether the expander is initially expanded or collapsed. Only intended with account access. Defaults to True.
        delete_buttons (bool, optional): Whether to display delete buttons for Pinecone and Chroma DB indexes. Defaults to False.
        set_secrets (bool, optional): Whether to set the secrets. Defaults to False.
    """
    with st.expander("Connection Status", expanded=expanded):
        # Set secrets and assign to environment variables
        if set_secrets:
            st.markdown("**Set API keys**:")
            keys={}
            # OPENAI_API_KEY
            keys['OPENAI_API_KEY'] = st.text_input('OpenAI API Key', type='password',help='OpenAI API Key: https://platform.openai.com/api-keys')
            if keys['OPENAI_API_KEY']!='':
                os.environ['OPENAI_API_KEY'] = keys['OPENAI_API_KEY'] 
            
            # VOYAGE_API_KEY
            keys['VOYAGE_API_KEY'] = st.text_input('Voyage API Key', type='password',help='Voyage API Key: https://dash.voyageai.com/api-keys')
            if keys['VOYAGE_API_KEY']!='':
                os.environ['VOYAGE_API_KEY'] = keys['VOYAGE_API_KEY']
            
            # PINECONE_API_KEY
            keys['PINECONE_API_KEY']=st.text_input('Pinecone API Key',type='password',help='Pinecone API Key: https://www.pinecone.io/')
            if keys['PINECONE_API_KEY']!='':
                os.environ['PINECONE_API_KEY'] = keys['PINECONE_API_KEY']

            # HUGGINGFACEHUB_API_TOKEN
            keys['HUGGINGFACEHUB_API_TOKEN'] = st.text_input('Hugging Face API Key', type='password',help='Hugging Face API Key: https://huggingface.co/settings/tokens')
            if keys['HUGGINGFACEHUB_API_TOKEN']!='':
                os.environ['HUGGINGFACEHUB_API_TOKEN'] = keys['HUGGINGFACEHUB_API_TOKEN']
            
            # LOCAL_DB_PATH
            keys['LOCAL_DB_PATH'] = st.text_input('Update Local Database Path',os.getenv('LOCAL_DB_PATH'),help='Path to local database (e.g. chroma)')
            if keys['LOCAL_DB_PATH']!='':
                os.environ['LOCAL_DB_PATH'] = keys['LOCAL_DB_PATH']

        if os.getenv('LOCAL_DB_PATH') is None:
            raise SecretKeyException('Local Database Path is required. Use an absolute path.','LOCAL_DB_PATH_MISSING')
        else:
            db_folder_path=os.getenv('LOCAL_DB_PATH')

        # Show key status
        st.markdown("**API key status** (Indicates status of local variable. It does not guarantee the key itself is correct):")
        st.markdown(test_key_status())

        # TODO the try statements below sometimes allow delete to be pressed but the index which does exist isn't deleted. Update so an error is thrown if the index is not deleted.
        # Pinecone
        st.markdown(show_pinecone_indexes())
        try:
            pinecone_indexes = [obj.name for obj in show_pinecone_indexes(format=False)['message']]
            if delete_buttons:
                pinecone_index_name = st.selectbox('Pinecone index to delete', pinecone_indexes)
                if st.button('Delete Pinecone index', help='This is permanent!'):
                    if pinecone_index_name.endswith("parent-child"):
                        rag_type = "Parent-Child"
                    elif pinecone_index_name.endswith("summary"):
                        rag_type = "Summary"
                    else:
                        rag_type = "Standard"
                    data_processing.delete_index('Pinecone', pinecone_index_name, rag_type, local_db_path=db_folder_path)
                    st.markdown(f"Index {pinecone_index_name} has been deleted.")
        except:
            pass

        # Chroma DB
        st.markdown(show_chroma_collections())
        try:
            chroma_db_collections = [obj.name for obj in show_chroma_collections(format=False)['message']]
            if delete_buttons:
                chroma_db_name = st.selectbox('Chroma database to delete', chroma_db_collections)
                if st.button('Delete Chroma database', help='This is permanent!'):
                    if chroma_db_name.endswith("parent-child"):
                        rag_type = "Parent-Child"
                    elif "-summary-" in chroma_db_name:
                        rag_type = "Summary"
                    else:
                        rag_type = "Standard"
                    data_processing.delete_index('ChromaDB', chroma_db_name, rag_type, local_db_path=db_folder_path)
                    st.markdown(f"Database {chroma_db_name} has been deleted.")
        except:
            pass
        
        # Ragatouille
        st.markdown(show_ragatouille_indexes())
        try:
            ragatouille_index_data=show_ragatouille_indexes(format=False)
            ragatouille_indexes = [obj for obj in ragatouille_index_data['message']]
            if delete_buttons and ragatouille_index_data['status']==True:
                ragatouille_name = st.selectbox('RAGatouille database to delete', ragatouille_indexes)
                if st.button('Delete RAGatouille database', help='This is permanent!'):
                    data_processing.delete_index('RAGatouille', ragatouille_name, "Standard", local_db_path=db_folder_path)
                    st.markdown(f"Index {ragatouille_name} has been deleted.")
        except:
            pass

        # Local database path
        st.markdown(f"Local database path: {os.environ['LOCAL_DB_PATH']}")
def st_setup_page(page_title: str, home_dir:str, sidebar_config: dict = None):
    """
    Sets up the Streamlit page with the given title and loads the sidebar configuration.

    Args:
        page_title (str): The title of the Streamlit page.
        home_dir (str): The path to the home directory.
        sidebar_config (dict, optional): The sidebar configuration. Defaults to None.

    Returns:
        tuple: A tuple containing the following:
            - paths (dict): A dictionary containing the following directory paths:
                - base_folder_path (str): The path to the root folder.
                - config_folder_path (str): The path to the config folder.
                - data_folder_path (str): The path to the data folder.
                - db_folder_path (str): The path to the database folder.
            - sb (dict): The sidebar configuration.
            - secrets (dict): A dictionary containing the set API keys.

    Raises:
        SecretKeyException: If there is an issue with the secret keys.

    """
    load_dotenv(find_dotenv(), override=True)

    base_folder_path = home_dir
    config_folder_path=os.path.join(base_folder_path, 'config')
    data_folder_path=os.path.join(base_folder_path, 'data')

    # Set the page title
    st.set_page_config(
        page_title=page_title,
        layout='wide'
    )
    st.title(page_title)

    # Set local database
    # Only show the text input if no value has been entered yet
    if not os.environ.get('LOCAL_DB_PATH'):
        local_db_path_input = st.empty()  # Create a placeholder for the text input
        warn_db_path=st.warning('Local Database Path is required to initialize. Use an absolute path.')
        local_db_path = local_db_path_input.text_input('Update Local Database Path', help='Path to local database (e.g. chroma).')
        if local_db_path:
            os.environ['LOCAL_DB_PATH'] = local_db_path
        else:
            st.stop()
    if os.environ.get('LOCAL_DB_PATH'): # If a value has been entered, update the environment variable and clear the text input
        try:
            local_db_path_input.empty()  # This will remove the text input from the page if it exists
            warn_db_path.empty()
        except:
            pass    # If the text input has already been removed, do nothing

    # Load sidebar
    try:
        if sidebar_config is None:
            sb=load_sidebar(os.path.join(config_folder_path,'config.json'))
        else:
            sb=load_sidebar(os.path.join(config_folder_path,'config.json'),
                            **sidebar_config)
    except SecretKeyException as e:
        # If no .env file is found, set the local db path when the warning is raised.
        st.warning(f"{e}")
        st.stop()
    try:
        secrets=set_secrets(sb) # Take secrets from .env file first, otherwise from sidebar
    except SecretKeyException as e:
        st.warning(f"{e}")
        st.stop()

    # Set db folder path based on env variable
    db_folder_path=os.environ['LOCAL_DB_PATH']

    paths={'base_folder_path':base_folder_path,
           'config_folder_path':config_folder_path,
           'data_folder_path':data_folder_path,
           'db_folder_path':db_folder_path}

    return paths,sb,secrets
def _format_key_status(key_status: str):
    """
    Formats the key status dictionary into a formatted string.

    Args:
        key_status (str): The dictionary containing the key status information.

    Returns:
        str: The formatted string representing the key status.
    """
    formatted_status = ""
    for key, value in key_status.items():
        status = value['status']
        if status:
            formatted_status += f"- {key}: :heavy_check_mark:\n"
        else:
            formatted_status += f"- {key}: :x:\n"
    return formatted_status
def _format_pinecone_status(pinecone_status):
    """
    Formats the Pinecone status into a markdown string.

    Args:
        pinecone_status (dict): The Pinecone status dictionary.

    Returns:
        str: The formatted markdown string.
    """
    index_description=''
    if pinecone_status['status']:
        for index in pinecone_status['message']:
            name = index['name']
            state = index['status']['state']
            status = ":heavy_check_mark:"
            index_description += f"- {name}: {state} ({status})\n"
        markdown_string = f"**Pinecone Indexes**\n{index_description}"
    else:
        message = pinecone_status['message']
        markdown_string = f"**Pinecone Indexes**\n- {message} :x:"
    return markdown_string
def _format_chroma_status(chroma_status):
    """
    Formats the chroma status dictionary into a markdown string.

    Args:
        chroma_status (dict): The chroma status dictionary containing the status and message.

    Returns:
        str: The formatted markdown string.
    """
    collection_description=''
    if chroma_status['status']:
        for index in chroma_status['message']:
            name = index.name
            status = ":heavy_check_mark:"
            collection_description += f"- {name}: ({status})\n"
        markdown_string = f"**ChromaDB Collections**\n{collection_description}"
    else:
        message = chroma_status['message']
        markdown_string = f"**ChromaDB Collections**\n- {message} :x:"
    return markdown_string
def _format_ragatouille_status(ragatouille_status):
    """
    Formats the ragatouille status for display.

    Args:
        ragatouille_status (dict): The ragatouille status dictionary.

    Returns:
        str: A formatted string representation of the ragatouille status.

    """
    index_description=''
    if ragatouille_status['status']:
        for index in ragatouille_status['message']:
            name = index
            status = ":heavy_check_mark:"
            index_description += f"- {name}: ({status})\n"
        markdown_string = f"**Ragatouille Indexes**\n{index_description}"
    else:
        message = ragatouille_status['message']
        markdown_string = f"**Ragatouille Indexes**\n- {message} :x:"
    return markdown_string