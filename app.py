import streamlit as st
import time
import tempfile
import os
from langchain_community.vectorstores import FAISS


# Set page config
st.set_page_config(page_title="Cat Expert", layout="wide")

st.title("🐾 An App all about Cats 😺")
tabs = st.tabs(["Chatbot", "Quiz", "Analyze Cat Food"])


with tabs[0]:
    from cat_chatbot import run_chatbot
    run_chatbot()

with tabs[1]:
    from cat_quiz import run_quiz
    run_quiz()

with tabs[2]:
    from ingredient_analysis import run_analyze_ingredients
    run_analyze_ingredients()