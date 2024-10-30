import streamlit as st
import os
from dotenv import load_dotenv,find_dotenv

load_dotenv(find_dotenv(), override=True)

# Get the config of the app from the environment variable, setup and run page
nav_bar=os.getenv('AEROSPACE_CHATBOT_CONFIG')
tool_dir="tools"

if nav_bar=="admin":
    pages=[st.Page(os.path.join(tool_dir,"Aerospace_Chatbot.py"),icon='🚀'),
           st.Page(os.path.join(tool_dir,"Database_Processing.py"),icon='📓'),
           st.Page(os.path.join(tool_dir,"Visualize_Data.py"),icon='📈')]
    st.session_state.config_file=os.path.join('../','config','config_admin.json')
elif nav_bar=="tester":
    pages=[st.Page(os.path.join(tool_dir,"Aerospace_Chatbot.py"),icon='🚀')]
    st.session_state.config_file=os.path.join('../','config','config_tester.json')
pg=st.navigation(pages)

st.set_page_config(
    page_title="Aerospace Chatbot",
    layout='wide',
    page_icon='🚀'
)
pg.run()