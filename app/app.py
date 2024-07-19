import streamlit as st
import os
from dotenv import load_dotenv,find_dotenv

load_dotenv(find_dotenv(), override=True)

# Get the config of the app from the environment variable, setup and run page
nav_bar=os.getenv('AEROSPACE_CHATBOT_CONFIG')
print(nav_bar)
tool_dir="tools"

if nav_bar=="admin":
    pages=[st.Page(os.path.join(tool_dir,"0_Home.py")),
           st.Page(os.path.join(tool_dir,"1_Database_Processing.py")),
           st.Page(os.path.join(tool_dir,"2_Chatbot.py")),
           st.Page(os.path.join(tool_dir,"3_Visualize_Data.py"))]
    st.session_state.config_file=os.path.join('../','config','config_admin.json')
elif nav_bar=="tester":
    pages=[st.Page(os.path.join(tool_dir,"0_Home.py")),
           st.Page(os.path.join(tool_dir,"2_Chatbot.py"))]
    st.session_state.config_file=os.path.join('../','config','config_tester.json')
pg=st.navigation(pages)

st.set_page_config(
    page_title="Aerospace Chatbot",
    layout='wide'
)
pg.run()