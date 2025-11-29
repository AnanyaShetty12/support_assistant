# app.py
import os
import uuid
from datetime import datetime
from pathlib import Path
import streamlit as st

import pandas as pd
from openai import OpenAI


OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
client = OpenAI(api_key=OPENAI_API_KEY)



# Debug (optional)
# st.write("KEY LOADED:", OPENAI_API_KEY is not None)

# ---- OpenAI Client ----
client = OpenAI(api_key=OPENAI_API_KEY)

# ---- CSV Paths ----
BASE_DIR = Path(__file__).parent
TICKETS_PATH = BASE_DIR / "tickets.csv"
FEEDBACK_PATH = BASE_DIR / "feedback.csv"

# ---- Page Config ----
st.set_page_config(page_title="AI Support Assistant", page_icon="🤖", layout="centered")

# ---- Dark UI ----
st.markdown("""
<style>
body {
    background: linear-gradient(135deg, #1f1f1f, #2c3e50);
    color: #f0f0f0;
}
.stButton>button {
    background-color: #3498db;
    color: white;
    border-radius: 10px;
    padding: 10px 20px;
    border: none;
    transition: transform 0.2s;
}
.stButton>button:hover {
    background-color: #2980b9;
    transform: scale(1.05);
}
.stTextInput>div>div>input, .stTextArea>div>div>textarea {
    background-color: #1f2a38;
    color: #f0f0f0;
    border-radius: 8px;
    padding: 10px;
}
.chat-bubble-user {
    background-color: #3498db;
    color: white;
    padding: 10px;
    border-radius: 15px 15px 0 15px;
    margin: 5px 0;
    max-width: 70%;
}
.chat-bubble-bot {
    background-color: #555;
    color: white;
    padding: 10px;
    border-radius: 15px 15px 15px 0;
    margin: 5px 0;
    max-width: 70%;
}
</style>
""", unsafe_allow_html=True)

# ---- Chat Bubbles ----
def user_message(msg):
    st.markdown(f'<div class="chat-bubble-user">{msg}</div>', unsafe_allow_html=True)

def bot_message(msg):
    st.markdown(f'<div class="chat-bubble-bot">{msg}</div>', unsafe_allow_html=True)

# ---- OpenAI Answer ----
def answer_with_openai(question):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a friendly support assistant."},
            {"role": "user", "content": question}
        ],
        temperature=0.3,
        max_tokens=400
    )
    return response.choices[0].message.content.strip()

# ---- Ticket System ----
def write_ticket(user_name, user_email, question):
    ticket_id = str(uuid.uuid4())[:8]
    now = datetime.utcnow().isoformat()

    ticket = {
        "ticket_id": ticket_id,
        "timestamp": now,
        "user_name": user_name,
        "user_email": user_email,
        "question": question,
        "status": "open"
    }

    df = pd.DataFrame([ticket])

    if TICKETS_PATH.exists():
        df.to_csv(TICKETS_PATH, mode="a", header=False, index=False)
    else:
        df.to_csv(TICKETS_PATH, index=False)

    return ticket_id

# ---- Feedback ----
def save_feedback(ref, rating, comment):
    fb = pd.DataFrame([{
        "id": str(uuid.uuid4())[:8],
        "timestamp": datetime.utcnow().isoformat(),
        "ref": ref,
        "rating": rating,
        "comment": comment
    }])

    if FEEDBACK_PATH.exists():
        fb.to_csv(FEEDBACK_PATH, mode="a", header=False, index=False)
    else:
        fb.to_csv(FEEDBACK_PATH, index=False)

# ---- UI ----
st.title("💬 AI Support Assistant")
st.write("Ask your question and get instant support")

st.sidebar.header("User Details")
user_name = st.sidebar.text_input("Your name", value="Guest")
user_email = st.sidebar.text_input("Your email (optional)")

query = st.text_area("Enter your question:", height=120)

if st.button("Submit"):
    if query.strip() == "":
        st.warning("Please enter a question.")
    else:
        user_message(query)

        try:
            answer = answer_with_openai(query)
            bot_message(answer)

            ticket_id = write_ticket(user_name, user_email, query)
            st.success(f"Ticket created successfully! Reference: **{ticket_id}**")

        except Exception as e:
            st.error(f"Error: {e}")

# ---- Feedback ----
st.subheader("⭐ Feedback")
ref = st.text_input("Ticket / Chat Reference (optional)")
rating = st.slider("Rate your experience", 1, 5, 4)
comment = st.text_area("Feedback message")

if st.button("Submit Feedback"):
    save_feedback(ref, rating, comment)
    st.success("Thank you! Your feedback has been recorded.")
