# app_dark_ui.py
import os
import uuid
from datetime import datetime
from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv
import openai

st.set_page_config(page_title="Support Assistant – FAQ Resolver", page_icon="💬")

# Load .env if present
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if OPENAI_API_KEY:
    openai.api_key = OPENAI_API_KEY

BASE_DIR = Path(__file__).parent
FAQ_PATH = BASE_DIR / "faqs.csv"
TICKETS_PATH = BASE_DIR / "tickets.csv"
FEEDBACK_PATH = BASE_DIR / "feedback.csv"

EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
TOP_K = 3
CONFIDENCE_THRESHOLD = 0.6

# --- Inject dark theme CSS ---
st.set_page_config(page_title="Support Assistant", page_icon="🤖", layout="centered")
st.markdown(
    """
    <style>
    body {
        background: linear-gradient(135deg, #1f1f1f, #2c3e50);
        color: #f0f0f0;
        font-family: 'Roboto', sans-serif;
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
    """, unsafe_allow_html=True
)

# --- Helper functions for chat bubbles ---
def user_message(msg):
    st.markdown(f'<div class="chat-bubble-user">{msg}</div>', unsafe_allow_html=True)

def bot_message(msg):
    st.markdown(f'<div class="chat-bubble-bot">{msg}</div>', unsafe_allow_html=True)

# --- Load FAQ model and embeddings ---
@st.cache_resource(show_spinner=False)
def load_faqs_and_embeddings(faq_path=FAQ_PATH, model_name=EMBED_MODEL_NAME):
    df = pd.read_csv(faq_path)
    df['question'] = df['question'].fillna('')
    df['answer'] = df['answer'].fillna('')
    model = SentenceTransformer(model_name)
    texts = (df['question'] + " " + df.get('category', '').fillna('')).tolist()
    embeddings = model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
    return df, model, embeddings

def semantic_search(query, df, model, embeddings, top_k=TOP_K):
    q_emb = model.encode([query], convert_to_numpy=True)
    sims = cosine_similarity(q_emb, embeddings)[0]
    top_idxs = np.argsort(sims)[::-1][:top_k]
    results = []
    for idx in top_idxs:
        results.append({
            "id": df.iloc[idx]['id'],
            "question": df.iloc[idx]['question'],
            "answer": df.iloc[idx]['answer'],
            "category": df.iloc[idx].get('category', ''),
            "score": float(sims[idx])
        })
    return results

def maybe_refine_with_openai(question, retrieved, openai_api_key=None):
    if not openai_api_key:
        return retrieved[0]['answer'], retrieved
    context_text = "\n\n".join([f"Q: {r['question']}\nA: {r['answer']}" for r in retrieved])
    prompt = (
        "You are a helpful customer support assistant. Use the following FAQ context to answer the user's question.\n\n"
        f"FAQ CONTEXT:\n{context_text}\n\nUser question: {question}\n\n"
        "Write a concise, friendly answer. If not covered, ask for more info and suggest escalation."
    )
    try:
        resp = openai.ChatCompletion.create(
            model="gpt-4o-mini" if hasattr(openai, "gpt") else "gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=300,
            temperature=0.2,
        )
        content = resp["choices"][0]["message"]["content"].strip() if "choices" in resp else str(resp)
        return content, retrieved
    except Exception:
        return retrieved[0]['answer'], retrieved

def write_ticket(user_name, user_email, question, context_snippets):
    ticket_id = str(uuid.uuid4())[:8]
    now = datetime.utcnow().isoformat()
    ticket = {
        "ticket_id": ticket_id,
        "timestamp": now,
        "user_name": user_name,
        "user_email": user_email,
        "question": question,
        "context": " | ".join(context_snippets),
        "status": "open"
    }
    df_ticket = pd.DataFrame([ticket])
    if TICKETS_PATH.exists():
        df_ticket.to_csv(TICKETS_PATH, mode='a', header=False, index=False)
    else:
        df_ticket.to_csv(TICKETS_PATH, index=False)
    return ticket_id

def save_feedback(ticket_or_query, rating, comment):
    now = datetime.utcnow().isoformat()
    fb = {"id": str(uuid.uuid4())[:8], "timestamp": now, "ref": ticket_or_query, "rating": rating, "comment": comment}
    df_fb = pd.DataFrame([fb])
    if FEEDBACK_PATH.exists():
        df_fb.to_csv(FEEDBACK_PATH, mode='a', header=False, index=False)
    else:
        df_fb.to_csv(FEEDBACK_PATH, index=False)

# --- Streamlit UI ---
st.title("💬 Support Assistant – FAQ Resolver & Escalation")
st.write("Ask your query and receive instant support")

# Load model and embeddings
with st.spinner("Loading knowledge base and model..."):
    faq_df, embed_model, faq_embeddings = load_faqs_and_embeddings()

# Sidebar for user info
st.sidebar.header("User Details")
user_name = st.sidebar.text_input("Your name", value="Guest")
user_email = st.sidebar.text_input("Your email (optional)")
use_openai = st.sidebar.checkbox("Use OpenAI to refine answers") if OPENAI_API_KEY else False

st.markdown("### Ask a Question")
query = st.text_area("Type your question here", height=100)

col1, col2 = st.columns([1,3])
with col1:
    topk = st.number_input("Top-K results", min_value=1, max_value=10, value=TOP_K, step=1)
with col2:
    threshold = st.slider("Confidence threshold", 0.0, 1.0, float(CONFIDENCE_THRESHOLD))

if st.button("Get Answer"):
    if not query.strip():
        st.warning("Please type a question.")
    else:
        with st.spinner("Searching FAQs..."):
            results = semantic_search(query, faq_df, embed_model, faq_embeddings, top_k=topk)
        best_score = results[0]['score']

        user_message(query)
        bot_message(f"Top match confidence: {best_score:.3f}")

        # Show top FAQ context
        with st.expander("Top retrieved FAQ context"):
            for r in results:
                st.markdown(f"- **Q:** {r['question']}\n- **A:** {r['answer']}\n- **Score:** {r['score']:.3f}")

        # Decide answer
        if best_score >= threshold:
            answer_text, used_context = maybe_refine_with_openai(query, results, openai_api_key=OPENAI_API_KEY if use_openai else None) if use_openai else (results[0]['answer'], results)
            bot_message(f"Suggested answer:\n{answer_text}")

            # Feedback buttons
            st.markdown("---")
            st.write("Was this answer helpful?")
            col_a, col_b, col_c = st.columns(3)
            if col_a.button("👍 Yes"):
                save_feedback(query, rating=5, comment="Helpful")
                st.info("Thanks for the feedback!")
            if col_b.button("🤷‍ No"):
                save_feedback(query, rating=2, comment="Not helpful")
                st.info("Feedback recorded. You can escalate the issue below.")
            if col_c.button("✍️ Leave comment"):
                comment = st.text_input("Leave a comment")
                if st.button("Submit comment"):
                    save_feedback(query, rating=3, comment=comment)
                    st.info("Comment saved!")

            # Escalation
            st.markdown("**Still need help?**")
            if st.button("Escalate to human support"):
                ticket_id = write_ticket(user_name, user_email, query, [r['question'] for r in used_context])
                st.warning(f"Escalated — ticket created (ID: {ticket_id}). Support will contact you.")
        else:
            bot_message("I couldn't find a confident answer in the FAQ.")
            bot_message(f"Best attempt:\n{results[0]['answer']}")
            if st.button("Escalate now"):
                ticket_id = write_ticket(user_name, user_email, query, [r['question'] for r in results])
                st.warning(f"Escalated — ticket created (ID: {ticket_id}). Support will contact you.")


            st.markdown("""
<hr>
<center>
📌 Support Assistant – FAQ Resolver & Escalation<br>
Developed using Streamlit & AI
</center>
""", unsafe_allow_html=True)
