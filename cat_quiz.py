import json
import random
import streamlit as st

@st.cache_resource
def load_quiz_data(path="cat_quiz_questions.json"):
    # Load the entire JSON file as a valid list of questions
    with open(path, "r", encoding="utf-8") as f:
        questions = json.load(f)  
    return questions

def run_quiz():
    st.markdown("""
        <div style='text-align: center;'>
            <h1>😺 Test your knowlege about cat</h1>
        </div>
        """, unsafe_allow_html=True)   

    questions = load_quiz_data()
    questions_per_set = 5

   # Initialize quiz state only once or on reset
    if "quiz_questions" not in st.session_state or st.session_state.get("quiz_reset", False):
        st.session_state.quiz_questions = random.sample(questions, questions_per_set)
        st.session_state.quiz_answers = [None] * questions_per_set
        st.session_state.quiz_submitted = False
        st.session_state.quiz_reset = False  # clear the reset trigger

    selected_questions = st.session_state.quiz_questions

    # Submit button
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        submitted = st.button("✅ Submit Quiz")
    # Immediately trigger feedback display
    if submitted:
        st.session_state.quiz_submitted = True

    show_feedback = submitted or st.session_state.get("quiz_submitted", False)

    # Show each question
    for i, q in enumerate(selected_questions):
        col1, col2, col3 = st.columns([1, 2, 1])

        with col2:
    
            with st.container():
                st.markdown("<div style='margin-top: 10px;'></div>", unsafe_allow_html=True)

                # Question
                st.markdown(
                    f"<div style='margin-bottom: 1px; font-weight: bold;'>Q{i+1}: {q['question']}</div>",
                    unsafe_allow_html=True
                )

                # Radio (no default selection)
                st.session_state.quiz_answers[i] = st.radio(
                    label=" ",
                    options=q["options"],
                    index=None if st.session_state.quiz_answers[i] is None else q["options"].index(st.session_state.quiz_answers[i]),
                    key=f"quiz_q_{i}"
                )

                # Show feedback after submission
                if show_feedback:
                    user_answer = st.session_state.quiz_answers[i]
                    correct_answer = q["answer"]

                    if user_answer == correct_answer:
                        st.success("✅ You are correct!")
                    else:
                        st.error(f"❌ Incorrect. The correct answer is **{correct_answer}**.")

        # Show final score and reset option
        if show_feedback:
            score = sum([
                1 for i, q in enumerate(selected_questions)
                if st.session_state.quiz_answers[i] == q["answer"]
            ])
            st.success(f"🎉 You scored {score} out of {questions_per_set}!")

            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                if st.button("🔄 Try Another Set"):
                    st.session_state.quiz_reset = True
                    st.rerun()
