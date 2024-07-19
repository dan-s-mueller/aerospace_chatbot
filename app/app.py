import streamlit as st
import os

# Get the config of the app from the environment variable, setup and run page
nav_bar=os.getenv('AEROSPACE_CHATBOT_CONFIG')
tool_dir="tools"
if nav_bar=="admin":
    pages=[st.Page(os.path.join(tool_dir,"0_Home.py")),
           st.Page(os.path.join(tool_dir,"1_Database_Processing.py")),
           st.Page(os.path.join(tool_dir,"2_Chatbot.py")),
           st.Page(os.path.join(tool_dir,"3_Visualize_Data.py"))]
elif nav_bar=="tester":
    pages=[st.Page(os.path.join(tool_dir,"0_Home.py")),
           st.Page(os.path.join(tool_dir,"2_Chatbot.py"))]

pg=st.navigation(pages)

st.set_page_config(
    page_title="Aerospace Chatbot",
    layout='wide'
)
pg.run()