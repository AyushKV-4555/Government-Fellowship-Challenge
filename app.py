import streamlit as st
import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# -----------------------------
# Load saved model and data
# -----------------------------
@st.cache_resource
def load_model_and_data():
    model_path = r"C:\DATA SCIENCE\govenment_intern\models\sbert_model"
    index_path = r"C:\DATA SCIENCE\govenment_intern\models\rag_index.faiss"
    data_path = r"C:\DATA SCIENCE\govenment_intern\models\rag_data.csv"

    # Load SentenceTransformer model
    model = SentenceTransformer(model_path)

    # Load FAISS index
    index = faiss.read_index(index_path)

    # Load question-answer data
    df = pd.read_csv(data_path)
    questions = df["question"].astype(str).tolist()
    answers = df["answer"].astype(str).tolist()

    return model, index, questions, answers

# Load everything
st.title("AI Chatbot (Project Samarth)")
st.write("Ask your question related to the dataset below")

with st.spinner("Loading model... Please wait"):
    model, index, questions, answers = load_model_and_data()

# -----------------------------
# Define answer retrieval function
# -----------------------------
def retrieve_answer(query, top_k=3):
    query_vec = model.encode([query])
    distances, indices = index.search(np.array(query_vec, dtype=np.float32), top_k)
    results = []
    for i, idx in enumerate(indices[0]):
        q = questions[idx]
        a = answers[idx]
        score = float(distances[0][i])
        results.append((q, a, score))
    return results

# -----------------------------
# Streamlit UI
# -----------------------------
query = st.text_input("🔍 Enter your question:")

if st.button("Get Answer"):
    if query.strip() == "":
        st.warning("Please enter a question!")
    else:
        results = retrieve_answer(query, top_k=3)
        best_answer = results[0][1]

        st.subheader("Answer:")
        st.write(best_answer)
        st.markdown("---")
