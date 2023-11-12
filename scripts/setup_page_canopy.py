"""
Page setup and execution for the aerospace mechanism chatbot
Example :        
-What types of lubricants are to be avoided when designing space mechanisms?      
-Follow up: Can you speak to what failures have occurred when using Perf luoropolyethers (PFPE)?          
"""

# COMMENT OUT IF IN IDE
import databutton as db

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

from tqdm.auto import tqdm
from typing import Tuple

from canopy.tokenizer import Tokenizer
from canopy.knowledge_base import KnowledgeBase
from canopy.context_engine import ContextEngine
from canopy.chat_engine import ChatEngine
from canopy.llm.openai import OpenAILLM
from canopy.llm.models import ModelParams
from canopy.models.data_models import Document, Messages, UserMessage, AssistantMessage
from canopy.models.api_models import ChatResponse

import openai

from IPython.display import display, Markdown

def chat(new_message: str, history: Messages) -> Tuple[str, Messages, ChatResponse]:
    messages = history + [UserMessage(content=new_message)]
    response = chat_engine.chat(messages,model_params=model_params)
    assistant_response = response.choices[0].message.content
    return assistant_response, messages + [AssistantMessage(content=assistant_response)], response

# USE IF IN DATABUTTON
# OPENAI_API_KEY=db.secrets.get('OPENAI_API_KEY')
PINECONE_ENVIRONMENT=db.secrets.get('PINECONE_ENVIRONMENT')
PINECONE_API_KEY=db.secrets.get('PINECONE_API_KEY')
os.environ["PINECONE_ENVIRONMENT"] = PINECONE_ENVIRONMENT
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY

# USE IF IN IDE
# from dotenv import load_dotenv,find_dotenv
# load_dotenv(find_dotenv(),override=True)
# PINECONE_ENVIRONMENT=os.getenv('PINECONE_ENVIRONMENT')
# PINECONE_API_KEY=os.getenv('PINECONE_API_KEY')

# Set the page title
st.title("Aerospace Mechanisms Chatbot")
st.markdown("""
This chatbot will look up from all Aerospace Mechanism Symposia in the following location: https://github.com/dsmueller3760/aerospace_chatbot/tree/main/data/AMS
* Available models: https://platform.openai.com/docs/models
* Model parameters: https://platform.openai.com/docs/api-reference/chat/create
* Pinecone: https://docs.pinecone.io/docs/projects#api-keys
* OpenAI API: https://platform.openai.com/api-keys

## What's under the hood?
* Source code: 
* Uses pinecone canopy: https://www.pinecone.io/blog/canopy-rag-framework/
* Response time ~45 seconds per prompt

""")
st.markdown("---")

# Add a sidebar for input options
st.title("Input")
st.sidebar.title("Input Options")

# Add input fields in the sidebar
model_name=st.sidebar.selectbox("Model", ['gpt-3.5-turbo''gpt-3.5-turbo-16k','gpt-3.5-turbo','gpt-3.5-turbo-1106','gpt-4','gpt-4-32k'], index=1)
model_list={'gpt-3.5-turbo':4096,
            'gpt-3.5-turbo-16k':16385,
            'gpt-3.5-turbo-1106':16385, 
            'gpt-4':8192,
            'gpt-4-32k':32768}
temperature = st.sidebar.slider("Temperature", min_value=0.0, max_value=2.0, value=0.0, step=0.1)
n=None  # Not used. How many chat completion choices to generate for each input message.
top_p=None  # Not used. Only use this or temperature. Where the model considers the results of the tokens with top_p probability mass. So 0.1 means only the tokens comprising the top 10% probability mass are considered.

k=st.sidebar.number_input("Number document chunks per query", min_value=1, step=1, value=15)
output_level=st.sidebar.selectbox("Level of Output", ["Concise", "Detailed", "No Limit"], index=2)
max_prompt_tokens=model_list[model_name]

# Add a section for secret keys
st.sidebar.title("Secret Keys")
OPENAI_API_KEY = st.sidebar.text_input("OpenAI API Key", type="password")

# Pinecone info for future info
# PINECONE_API_KEY = st.sidebar.text_input("Pinecone API Key", type="password")
# PINECONE_ENVIRONMENT = st.sidebar.text_input("Pinecone Environment", type="password")
# index_name = st.sidebar.text_input("Pinecone Index Name",value="canopy--ams")
index_name='canopy--ams'

if OPENAI_API_KEY:
    # Set up chat history
    history = st.session_state.get("history",[])
    chat_history = st.session_state.get("chat_history", [])
    message_id = st.session_state.get("message_id", 0)
        
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
    openai.api_key = OPENAI_API_KEY
    
    # Create a text input field for the user query
    user_input = st.text_input("User Input", "",)
    
    # Disable the button until text_input is not empty
    if user_input:
        button_disabled = False
    else:
        button_disabled = True
    button_clicked = st.button("Send", disabled=button_disabled)
    
    status_placeholder = st.empty()

    # TODO: make this better with https://docs.streamlit.io/knowledge-base/tutorials/build-conversational-apps
    # Start prompting
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
                max_generated_tokens = 50
            elif output_level == "Detailed":
                max_generated_tokens = 516
            else:
                max_generated_tokens = None
    
            # Inialize canopy
            Tokenizer.initialize()
            kb = KnowledgeBase(index_name=index_name,
                               default_top_k=k)
            kb.connect()
            context_engine = ContextEngine(kb)
            llm=OpenAILLM(model_name=model_name)
            chat_engine = ChatEngine(context_engine,
                                     llm=llm,
                                     max_generated_tokens=max_generated_tokens,
                                     max_prompt_tokens=max_prompt_tokens)
            model_params=ModelParams(temperature=temperature,
                                     n=n,  # number of completions to generate
                                     top_p=top_p)
            
            # Prompt and get response
            response, history, chat_response = chat(user_input, history)
    
            # Add the user input and AI response to the chat history with message ID
            chat_history.append(f"AI: {response}")
            chat_history.append(f"User: {user_input}")
            chat_history.append(f"Message ID: {message_id}")
            
            # Add a horizontal line between messages
            chat_history.append("---")
            status_placeholder.text("Chat history updated.")
        else:
            status_placeholder.text("Please enter a prompt.")
        
    # Store the updated chat history and message ID in the session state
    st.session_state["chat_history"] = chat_history
    st.session_state["history"] = history
    st.session_state["message_id"] = message_id
    
    # Display the chat history in descending order
    for message in reversed(chat_history):
        st.markdown(message)
else:
    st.warning("No API key found. Add your API key in the sidebar under Secret Keys. Find it or create one here: https://platform.openai.com/api-keys")
    st.info("Your API-key is not stored in any form by this app. However, for transparency it is recommended to delete your API key once used.")