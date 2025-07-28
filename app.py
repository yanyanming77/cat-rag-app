import streamlit as st
import time
import tempfile
import os
from langchain_community.vectorstores import FAISS


# Set page config
st.set_page_config(page_title="Cat Expert", layout="wide")

# Sidebar for navigation
st.sidebar.title("🐾 Navigation")
page = st.sidebar.radio("Go to", ["Chatbot", "Quiz"])

# Render corresponding page
if page == "Chatbot":
    from cat_chatbot import run_chatbot
    run_chatbot()

elif page == "Quiz":
    from cat_quiz import run_quiz
    run_quiz()