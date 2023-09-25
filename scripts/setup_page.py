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

# Set the page title
st.title("Aerospace Mechanisms Chatbot")
st.markdown("""
This chatbot will look up from all Aerospace Mechanism Symposia in the following location: https://github.com/dsmueller3760/aerospace_chatbot/tree/main/data

Notes on usage:
* If you see an error message when you first open, put in your OpenAI key in the sidebar. See here for details: https://platform.openai.com/account/api-keys
* Pinecone API key and environment can be blank. Leave the Pinecone index name as the default.
""")
st.markdown("---")

# Add a sidebar for input options
st.title("Input")
st.sidebar.title("Input Options")

# Add input fields in the sidebar
output_level = st.sidebar.selectbox("Level of Output", ["Concise", "Detailed"], index=1)
k = st.sidebar.number_input("Number of items per prompt", min_value=1, step=1, value=6)
search_type = st.sidebar.selectbox("Search Type", ["similarity", "mmr"], index=1)
temperature = st.sidebar.slider("Temperature", min_value=0.0, max_value=1.0, value=0.0, step=0.1)
verbose = st.sidebar.checkbox("Verbose output")
chain_type = st.sidebar.selectbox("Chain Type", ["stuff", "map_reduce"], index=0)

# Add a section for secret keys
st.sidebar.title("Secret Keys")
openai_api_key = st.sidebar.text_input("OpenAI API Key", type="password")
pinecone_api_key = st.sidebar.text_input("Pinecone API Key", type="password")
pinecone_environment = st.sidebar.text_input("Pinecone Environment", type="password")
index_name = st.sidebar.text_input("Pinecone Index Name",value="langchain-quickstart")

# Instantiate openai and pinecone things, including keys
# TODO: update all of these to take from the webpage. Currently they pull from .env file.
load_dotenv(find_dotenv(),override=True)

embeddings_model = OpenAIEmbeddings(model="text-embedding-ada-002",openai_api_key=os.getenv("OPENAI_API_KEY"))

# Pinecone
index_name = 'langchain-quickstart'
pinecone.init(
    api_key=os.getenv('PINECONE_API_KEY'),
    environment=os.getenv('PINECONE_ENVIRONMENT'),
    index_name=index_name
)

# Add filter toggle
filter_toggle = st.checkbox("Filter response with last received sources?")

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

        llm = OpenAI(temperature=temperature,
                     openai_api_key=os.getenv("OPENAI_API_KEY"),
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