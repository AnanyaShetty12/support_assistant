🚀 Support Assistant – FAQ Resolver & Escalation
 Overview
The Support Assistant is an AI-powered web application that helps users quickly find answers to common support questions.
It automatically checks whether the user’s query matches any existing Frequently Asked Questions (FAQs).
If a suitable answer is found, it returns the best match. If no relevant FAQ exists, the system initiates escalation, indicating that the query needs human support intervention.

This agent reduces support workload by resolving repetitive queries while ensuring unresolved issues are escalated.

⭐ Features
Feature                       | Description
------------------------------|-----------------------------------------------------------------
FAQ-based question answering  | Finds the most relevant FAQ response using embedding similarity
Top-K question matching       | Ranks best results based on cosine similarity
Confidence threshold check    | Prevents wrong answers & triggers escalation
Escalation detection          | Marks non-answerable queries for human support
Web-based interface           | Built using Streamlit for instant interaction
Cloud deployment              | Hosted easily on Streamlit Cloud

🛠 Tech Stack & APIs Used
Component             | Technology
----------------------|--------------------
Frontend              | Streamlit
Backend               | Python
Embedding Model       | SentenceTransformer
Similarity Algorithm  | Cosine Similarity
Deployment            | Streamlit Cloud + GitHub

Setup & Run Instructions (Local Machine)

1️⃣ Clone the Repository
git clone https://github.com/AnanyaShetty12/support_assistant.git
cd support_assistant

2️⃣ Create Virtual Environment (Recommended)
python -m venv venv

Activate virtual environment:
Windows: venv\Scripts\activate

3️⃣ Install Required Libraries
pip install -r requirements.txt

4️⃣ Run the Application
streamlit run app.py

System Workflow 
1. User enters a question in Streamlit UI
2. Query is converted to embeddings
3. Embedding is compared with FAQ database via cosine similarity
4. If similarity ≥ threshold → display best match
5. If similarity < threshold → trigger escalation message

Potential Improvements
- Add escalation email or ticket creation
- Multilingual FAQ search
- Admin panel to update FAQs dynamically
- Integration with CRM / company database
- LLM (GPT) fallback for completely new queries

🔗 Demo Links
Live App: https://supportassistant-12.streamlit.app/
GitHub Repo: https://github.com/AnanyaShetty12/support_assistant

