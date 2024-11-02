import cProfile, pstats  # Add these imports
from pstats import SortKey

# Add profiler initialization right after imports
profiler = cProfile.Profile()
profiler.enable()

import streamlit as st
import os
from dotenv import load_dotenv,find_dotenv

# Set page config first to avoid re-renders
st.set_page_config(
    page_title="Aerospace Chatbot",
    layout='wide',
    page_icon='🚀'
)

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
pg.run()
# Add this at the end of the file, right before or after the reset button
profiler.disable()
# Sort stats by cumulative time
stats = pstats.Stats(profiler).sort_stats(SortKey.CUMULATIVE)
# Export stats to file in same directory as script
stats_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'profile_stats.txt')

# Redirect stdout to capture stats output
import io
output_stream = io.StringIO()
stats.stream = output_stream
stats.print_stats(50)  # Print top 20 stats

# Write captured output to file
with open(stats_file, 'w') as f:
    f.write(output_stream.getvalue())
