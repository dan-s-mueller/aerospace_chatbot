import streamlit as st
import openai
import secrets

# Set the page title
st.title("Aerospace Mechanisms Chatbot")
filter = st.checkbox("Filter response with last received sources?")

# Add a sidebar for input options
st.sidebar.title("Input Options")

# Add input fields in the sidebar
output_level = st.sidebar.selectbox("Level of Output", ["Concise", "To the Point", "Detailed"],index=2)
k = st.sidebar.number_input("Number of items per prompt", min_value=1, step=1, value=6)
search_type = st.sidebar.selectbox("Search Type", ["similarity", "mmr"],index=1)
temperature = st.sidebar.slider("Temperature", min_value=0.0, max_value=1.0, value=0.0, step=0.1)
verbose = st.sidebar.checkbox("Verbose output")
chain_type = st.sidebar.selectbox("Chain Type", ["stuff", "map_reduce"],index=0)

# Add a section for secret keys
st.sidebar.title("Secret Keys")
openai_api_key = st.sidebar.text_input("OpenAI API Key", type="password")
pinecone_api_key = st.sidebar.text_input("Pinecone API Key", type="password")
pinecone_environment = st.sidebar.text_input("Pinecone Environment", type="password")

def generate_response(input_text):
    # Replace this function with your chat model logic
    return "test"

chat_history = []
chat_history = st.session_state.get("chat_history", [])

# Create a text input field for the user query
user_input = st.text_input("User Input", "")

if st.button("Send"):
    if user_input:
        # Generate a response using your chat model
        ai_response = generate_response(user_input)
        
        # Add the user input and AI response to the chat history
        chat_history.append(f"AI: {ai_response}")
        chat_history.append(f"User: {user_input}")
        
        # Add a horizontal line between messages
        chat_history.append("---")
    
# Store the updated chat history in the session state
st.session_state["chat_history"] = chat_history

# Display the chat history in descending order
for message in reversed(chat_history):
    st.markdown(message)