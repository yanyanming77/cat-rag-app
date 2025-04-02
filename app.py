import streamlit as st
import time

st.set_page_config(page_title="Cat Expert Chatbot 🐾", layout="wide")
from rag_core import generate_answer, strip_html

st.markdown("""
    <style>
        .element-container {
            animation: fadeIn 0.3s ease-in;
        }
        @keyframes fadeIn {
            from { opacity: 0; transform: scale(0.95); }
            to { opacity: 1; transform: scale(1); }
        }

        /* Assistance message background color */
        .assistant-message {
        background-color: #fbedf5;
        padding: 12px;
        border-radius: 10px;
        color: black;
        font-size: 16px;
        }
    </style>
""", unsafe_allow_html=True)

# Initialize conversation history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Always show the title once
st.markdown("""
    <div style='text-align: center;'>
        <h1>😺 Ask the Cat Expert</h1>
        <div style='font-size: 18px; color: #555;'>
            Welcome to the <b>Cat Expert Chatbot Designed by Ming</b>! 🐾<br>
        </div>
        <div style="font-size: 10px; color: #555; font-style: italic;">
            Ask questions about cat health, behavior, grooming, nutrition, plants toxic to cats, or how to raise cats and newborns together — powered by ChatGPT-4o-mini!
        </div>
    </div>
    """, unsafe_allow_html=True)    

# Add info box on main screen if sidebar might be overlooked:
st.markdown(
    """
    <div style="text-align: center; font-size: 14px; color: #3B3B3B; background-color: #E1F5FE; 
                padding: 10px; border-left: 5px solid #2196F3; border-radius: 3px; margin-bottom: 30px;">
        <strong> 🧹Need to start over?</strong> Use the sidebar to reset the conversation.
    </div>
    """,
    unsafe_allow_html=True
)

# Reset button right under the title
with st.sidebar:
    st.header("Options")
    if st.button("🧹 Reset Conversation"):
        st.session_state.messages = []
        st.session_state.reset_done = True
        st.rerun()
    # Download button (simulated with conditional logic)
    if st.button("📥 Download Conversation"):
        if st.session_state.get("messages"):
            chat_text = "\n".join(
                [f"{m['role'].capitalize()}: {strip_html(m['content'])}" for m in st.session_state.messages]
            )
            st.download_button("Click to confirm download", chat_text, file_name="cat_chat.txt", key="download")
        else:
            st.warning("⚠️ The conversation is empty. Nothing to download.")

# Outside sidebar, after page reload show success message after reset
if st.session_state.get("reset_done", False):
    st.success("Conversation reset. You can start a new conversation")
    del st.session_state["reset_done"]

# Display chat history (exclude the *last* assistant message if a new one is being typed)
if not st.session_state.get("generating_response", False):
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"], unsafe_allow_html=True)

# User input box (chat-style)
if prompt := st.chat_input("Ask me anything about cats..."):
    st.session_state.generating_response = True

    # Show user's message
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate assistant response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                # get chat history
                history = st.session_state.messages[-6:] # 3 turns = 6 messages
                response = generate_answer(prompt, history)
            except Exception as e:
                response = "Oops! Something went wrong. Please try again later."
                print(f"error: {e}")

        formatted_response = f"""
            <div class="assistant-message">
                {response}
            </div>
        """
        st.markdown(formatted_response, unsafe_allow_html=True)

    # Save messages to session_state (already formatted!)
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.session_state.messages.append({"role": "assistant", "content": formatted_response})
    st.session_state.generating_response = False
    st.rerun() 
