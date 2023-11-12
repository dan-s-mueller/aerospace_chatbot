"""
Page setup and execution for the aerospace mechanism chatbot
Example :        
-What can you tell me about latch mechanism design failures which have occurred        
-Follow up: Which one of the sources discussed volatile spherical joint interfaces           
"""
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

# Set secrets
PINECONE_ENVIRONMENT=db.secrets.get('PINECONE_ENVIRONMENT')
PINECONE_API_KEY=db.secrets.get('PINECONE_API_KEY')

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
* Uses custom langchain functions with QA retrieval: https://js.langchain.com/docs/modules/chains/popular/chat_vector_db_legacy

## Notes on usage:
* Leave the Pinecone index name as the default.
""")
st.markdown("---")

# Add a sidebar for input options
st.title("Input")
st.sidebar.title("Input Options")

# Add input fields in the sidebar
output_level = st.sidebar.selectbox("Level of Output", ["Concise", "Detailed"], index=1)
k = st.sidebar.number_input("Number of items per prompt", min_value=1, step=1, value=4)
search_type = st.sidebar.selectbox("Search Type", ["similarity", "mmr"], index=1)
temperature = st.sidebar.slider("Temperature", min_value=0.0, max_value=2.0, value=0.0, step=0.1)
verbose = st.sidebar.checkbox("Verbose output")
chain_type = st.sidebar.selectbox("Chain Type", ["stuff", "map_reduce"], index=0)

# Add a section for secret keys
st.sidebar.title("Secret Keys")
OPENAI_API_KEY = st.sidebar.text_input("OpenAI API Key", type="password")

# COMMENT OUT BEFORE DISTRIBUTING
# OPENAI_API_KEY=db.secrets.get('OPENAI_API_KEY')

# PINECONE_API_KEY = st.sidebar.text_input("Pinecone API Key", type="password")
# PINECONE_ENVIRONMENT = st. sidebar.text_input("Pinecone Environment", type="password")
index_name = st.sidebar.text_input("Pinecone Index Name",value="canopy--ams")

# Pinecone
pinecone.init(
    api_key=PINECONE_API_KEY,
    environment=PINECONE_ENVIRONMENT,
    index_name=index_name
)

if OPENAI_API_KEY:
    openai.api_key = OPENAI_API_KEY
    embeddings_model = OpenAIEmbeddings(model="text-embedding-ada-002",openai_api_key=OPENAI_API_KEY)

    # Set up chat history
    qa_model_obj = st.session_state.get("qa_model_obj",[])
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

    # Add filter toggle
    filter_toggle = st.checkbox("Filter response with last received sources?")

    
    # Start prompting
    if button_clicked:
        if user_input:
            status_placeholder.text("Updating chat...")
            # Increment the message ID
            message_id += 1
            
            # Process some items
            if output_level == "Concise":
                out_token = 50
            else:
                out_token = 516

            # Define LLM parameters
            llm = OpenAI(temperature=temperature,
                         openai_api_key=OPENAI_API_KEY,
                         max_tokens=out_token)
            
            if message_id>1 and filter_toggle:
                filter_list = list(set(item["source"] for item in qa_model_obj.sources[-1]))
                filter_items=[]
                for item in filter_list:
                    filter_item={"source": item}
                    filter_items.append(filter_item)
                filter={"$or":filter_items}
            else:
                filter_items=None
                filter=None
            
            qa_model_obj=queries.QA_Model(index_name,
                                embeddings_model,
                                llm,
                                k,
                                search_type,
                                verbose,
                                filter=filter)
    
            # Generate a response using your chat model
            qa_model_obj.query_docs(user_input)
            ai_response=qa_model_obj.result['answer']
            
            # Add the user input and AI response to the chat history with message ID
            chat_history.append(f"References: {qa_model_obj.sources[-1]}")
            chat_history.append(f"AI: {ai_response}")
            chat_history.append(f"User: {user_input}")
            chat_history.append(f"Source filter: {filter_items}")
            chat_history.append(f"Message ID: {message_id}")
            
            # Add a horizontal line between messages
            chat_history.append("---")
            status_placeholder.text("Chat history updated.")
        else:
            status_placeholder.text("Please enter a prompt.")

    # Store the updated chat history and message ID in the session state
    st.session_state["qa_model_obj"] = qa_model_obj
    st.session_state["chat_history"] = chat_history
    st.session_state["message_id"] = message_id
    
    # Display the chat history in descending order
    for message in reversed(chat_history):
        st.markdown(message)
else:
    st.warning("No API key found. Add your API key in the sidebar under Secret Keys. Find it or create one here: https://platform.openai.com/api-keys")
    st.info("Your API-key is not stored in any form by this app. However, for transparency it is recommended to delete your API key once used.")