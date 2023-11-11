"""
Page setup and execution for the aerospace mechanism chatbot
Example :        
-What types of lubricants are to be avoided when designing space mechanisms?      
-Follow up: Can you speak to what failures have occurred when using Perf luoropolyethers (PFPE)?          
"""
import os
import glob
import data_import
import queries
import pinecone
from dotenv import load_dotenv, find_dotenv
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import OpenAI
import streamlit as st
import openai
import secrets

from canopy.tokenizer import Tokenizer
from canopy.knowledge_base import KnowledgeBase
from canopy.models.data_models import Document
from tqdm.auto import tqdm
from canopy.context_engine import ContextEngine
from canopy.chat_engine import ChatEngine
from typing import Tuple
from canopy.models.data_models import Messages, UserMessage, AssistantMessage
from IPython.display import display, Markdown

def chat(new_message: str, history: Messages) -> Tuple[str, Messages]:
    messages = history + [UserMessage(content=new_message)]
    response = chat_engine.chat(messages)
    assistant_response = response.choices[0].message.content
    return assistant_response, messages + [AssistantMessage(content=assistant_response)]

# Set secrets if on databutton
# # OPENAI_API_KEY=db.secrets.get('OPENAI_API_KEY')
# PINECONE_ENVIRONMENT=db.secrets.get('PINECONE_ENVIRONMENT_PERSONAL')
# # PINECONE_ENVIRONMENT=db.secrets.get('PINECONE_ENVIRONMENT_WORK')
# PINECONE_API_KEY=db.secrets.get('PINECONE_API_KEY_PERSONAL')
# # PINECONE_API_KEY=db.secrets.get('PINECONE_API_KEY_WORK')

# Use this if in IDE
from dotenv import load_dotenv,find_dotenv
load_dotenv(find_dotenv(),override=True)
PINECONE_ENVIRONMENT=os.getenv('PINECONE_ENVIRONMENT')
PINECONE_API_KEY=os.getenv('PINECONE_API_KEY')

# Set the page title
st.title("Aerospace Mechanisms Chatbot")
st.markdown("""
This chatbot will look up from all Aerospace Mechanism Symposia in the following location: https://github.com/dsmueller3760/aerospace_chatbot/tree/main/data/AMS

Notes on usage:
* If you see an error message when you first open, put in your OpenAI key in the sidebar. See here for details: https://platform.openai.com/account/api-keys
* Leave the Pinecone index name as the default.
""")
st.markdown("---")

# Add a sidebar for input options
st.title("Input")
st.sidebar.title("Input Options")

# Add input fields in the sidebar
output_level = st.sidebar.selectbox("Level of Output", ["Concise", "Detailed"], index=1)
k = st.sidebar.number_input("Number of items per prompt", min_value=1, step=1, value=4)
# search_type = st.sidebar.selectbox("Search Type", ["similarity", "mmr"], index=1)
temperature = st.sidebar.slider("Temperature", min_value=0.0, max_value=1.0, value=0.0, step=0.1)
# verbose = st.sidebar.checkbox("Verbose output")
# chain_type = st.sidebar.selectbox("Chain Type", ["stuff", "map_reduce"], index=0)

# Add a section for secret keys
st.sidebar.title("Secret Keys")
OPENAI_API_KEY = st.sidebar.text_input("OpenAI API Key", type="password")
# PINECONE_API_KEY = st.sidebar.text_input("Pinecone API Key", type="password")
# PINECONE_ENVIRONMENT = st.sidebar.text_input("Pinecone Environment", type="password")
index_name = st.sidebar.text_input("Pinecone Index Name",value="canopy--ams")

# DELETE ME
OPENAI_API_KEY=os.getenv('OPENAI_API_KEY')

# Instantiate openai and pinecone things, including keys
# TODO: update all of these to take from the webpage. Currently they pull from .env file.
# load_dotenv(find_dotenv(),override=True)

# embeddings_model = OpenAIEmbeddings(model="text-embedding-ada-002",openai_api_key=OPENAI_API_KEY)

# # Pinecone
# pinecone.init(
#     api_key=PINECONE_API_KEY,
#     environment=PINECONE_ENVIRONMENT,
#     index_name=index_name
# )

# Add filter toggle
# filter_toggle = st.checkbox("Filter response with last received sources?")

# Set up chat history
history = st.session_state.get("history",[])
chat_history = st.session_state.get("chat_history", [])
message_id = st.session_state.get("message_id", 0)

# Create a text input field for the user query
user_input = st.text_input("User Input", "",)

# Disable the button until text_input is not empty
if user_input:
    button_disabled = False
else:
    button_disabled = True
button_clicked = st.button("Send", disabled=button_disabled)

status_placeholder = st.empty()

if button_clicked:
    if user_input:
        status_placeholder.text("Updating chat...")

        # helpful debugging
        print('History:')
        print(history)
        print('Message ID: '+str(message_id))
        print('Chat history:')
        print(chat_history)

        # Increment the message ID
        message_id += 1
        
        # Process some items
        if output_level == "Concise":
            out_token = 50
        else:
            out_token = 516

        Tokenizer.initialize()
        # index_name='canopy--ams'
        kb = KnowledgeBase(index_name=index_name)
        kb.connect()
        context_engine = ContextEngine(kb)
        chat_engine = ChatEngine(context_engine)

        response, history = chat(user_input, history)
        # display(Markdown(response))

        llm = OpenAI(temperature=temperature,
                     openai_api_key=OPENAI_API_KEY,
                     max_tokens=out_token)
        
        # if message_id>1 and filter_toggle:
        #     filter_list = list(set(item["source"] for item in qa_model_obj.sources[-1]))
        #     filter_items=[]
        #     for item in filter_list:
        #         filter_item={"source": item}
        #         filter_items.append(filter_item)
        #     filter={"$or":filter_items}
        # else:
        #     filter_items=None
        #     filter=None
        
        # qa_model_obj=queries.QA_Model(index_name,
        #                     embeddings_model,
        #                     llm,
        #                     k,
        #                     search_type,
        #                     verbose,
        #                     filter=filter)

        # Generate a response using your chat model
        # qa_model_obj.query_docs(user_input)
        # ai_response=qa_model_obj.result['answer']
        
        # Add the user input and AI response to the chat history with message ID
        chat_history.append(f"AI: {response}")
        chat_history.append(f"User: {user_input}")
        chat_history.append(f"Message ID: {message_id}")

        # # Add the user input and AI response to the chat history with message ID
        # chat_history.append(f"References: {qa_model_obj.sources[-1]}")
        # chat_history.append(f"AI: {ai_response}")
        # chat_history.append(f"User: {user_input}")
        # chat_history.append(f"Source filter: {filter_items}")
        # chat_history.append(f"Message ID: {message_id}")
        
        # Add a horizontal line between messages
        chat_history.append("---")
        status_placeholder.text("Chat history updated.")
    else:
        status_placeholder.text("Please enter a prompt.")
    
# Store the updated chat history and message ID in the session state
# st.session_state["qa_model_obj"] = qa_model_obj
st.session_state["chat_history"] = chat_history
st.session_state["history"] = history
st.session_state["message_id"] = message_id

# Display the chat history in descending order
for message in reversed(chat_history):
    st.markdown(message)