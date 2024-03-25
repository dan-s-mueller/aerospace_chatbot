import os, sys, time, logging
from dotenv import load_dotenv,find_dotenv
import streamlit as st

from langchain_community.vectorstores import Pinecone
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import VoyageEmbeddings
from ragatouille import RAGPretrainedModel

sys.path.append('../../src/aerospace_chatbot')  # Add package to path
import admin, queries


def _reset_conversation():
    st.session_state.qa_model_obj = []
    st.session_state.message_id = 0
    st.session_state.messages = []
    return None

# # Read environment variables
# load_dotenv(find_dotenv(),override=True)

# # Define root path. All variables in this script are relative to the root path, not the run path.
# current_script_path = os.path.abspath(__file__) # Get the absolute path of the current script
# current_dir = os.path.dirname(current_script_path)  # Get the directory containing the current script
# base_folder_path = os.path.join(current_dir, '..', '..')    # Define the path to the root folder
# base_folder_path = os.path.normpath(base_folder_path)  # Normalize the path
# logging.info(f'Root folder path: {base_folder_path}')

# # Set the page title, load sidebar
# st.set_page_config(
#     page_title='Aerospace Chatbot',
#     layout='wide'
# )
# st.title('Aerospace Chatbot')
# sb=admin.load_sidebar(config_file=os.path.join(base_folder_path,'config','config.json'),
#                       index_data_file=os.path.join(base_folder_path,'config','index_data.json'),
#                       vector_databases=True,
#                       embeddings=True,
#                       rag_type=True,
#                       index_name=True,
#                       llm=True,
#                       model_options=True,
#                       secret_keys=True)
# try:
#     secrets=admin.set_secrets(sb) # Take secrets from .env file first, otherwise from sidebar
# except admin.SecretKeyException as e:
#     st.warning(f"{e}")
#     st.stop()

# # Define use case specific paths
# config_folder_path=os.path.join(current_dir, 'config')
# data_folder_path=os.path.join(base_folder_path, 'data')
# db_folder_path=os.path.join(base_folder_path, sb['keys']['LOCAL_DB_PATH'])
# logging.info(f'Config folder path: {config_folder_path}')
# logging.info(f'Data folder path: {data_folder_path}')
# logging.info(f'Database folder path: {db_folder_path}')

paths,sb,secrets=admin.st_setup_page('Aerospace Chatbot',
                                     {'vector_databases':True,
                                      'embeddings':True,
                                      'rag_type':True,
                                      'index_name':True,
                                      'llm':True,
                                      'model_options':True,
                                      'secret_keys':True})

# Add expander with functionality details.
with st.expander('''What's under the hood?'''):
    st.markdown('''
    This chatbot will look up from all Aerospace Mechanism Symposia in the following location: https://github.com/dsmueller3760/aerospace_chatbot/tree/main/data/AMS
    Example questions:
    * What are examples of latch failures which have occurred due to improper fitup?
    * What are examples of lubricants which should be avoided for space mechanism applications?
    * What can you tell me about the efficacy of lead naphthenate for wear protection in space mechanisms?
    ''')

# Set up chat history
if 'qa_model_obj' not in st.session_state:
    st.session_state.qa_model_obj = []
    logging.info('QA model object initialized.')
if 'message_id' not in st.session_state:
    st.session_state.message_id = 0
    logging.info('Message ID initialized.')
if 'messages' not in st.session_state:
    st.session_state.messages = []
    logging.info('Messages initialized.')
for message in st.session_state.messages:
    with st.chat_message(message['role']):
        st.markdown(message['content'])
    logging.info('Chat history loaded.')

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
            st.write('Starting reponse generation for message: '+str(st.session_state.message_id))
            
            if st.session_state.message_id==1:
                # Define embeddings
                if sb['query_model']=='Openai':
                    query_model=OpenAIEmbeddings(model=sb['embedding_name'],openai_api_key=secrets['OPENAI_API_KEY'])
                elif sb['query_model']=='Voyage':
                    query_model=VoyageEmbeddings(model=sb['embedding_name'],voyage_api_key=secrets['VOYAGE_API_KEY'])
                elif sb['index_type']=='RAGatouille':
                    query_model=RAGPretrainedModel.from_index(os.path.join(paths['db_folder_path'],'.ragatouille/colbert/indexes',sb['index_selected']))

                # Define LLM
                llm=admin.set_llm(sb,secrets,type='prompt')

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
                                                               filter_arg=False,
                                                               local_db_path=paths['db_folder_path'])
            if st.session_state.message_id>1:
                # Update LLM
                llm=admin.set_llm(sb,secrets,type='prompt')

                st.session_state.qa_model_obj.update_model(llm,
                                                           k=sb['model_options']['k'],
                                                           search_type=sb['model_options']['search_type'],
                                                           filter_arg=False)
            
            st.write('Searching vector database, generating prompt...')
            st.session_state.qa_model_obj.query_docs(prompt)
            ai_response=st.session_state.qa_model_obj.ai_response
            message_placeholder.markdown(ai_response)
            st.write("Alternative questions: \n\n\n"+
                     st.session_state.qa_model_obj.generate_alternative_questions(prompt,llm,response=ai_response))

            t_delta=time.time() - t_start
            status.update(label='Prompt generated in '+"{:10.3f}".format(t_delta)+' seconds', state='complete', expanded=False)
            
        st.session_state.messages.append({'role': 'assistant', 'content': ai_response})

# Add reset button
if st.button('Restart session'):
    _reset_conversation()
