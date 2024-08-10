import streamlit as st
import openai
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

openai.api_key = st.secrets["mykey"]

# Load the dataset
@st.cache_data
def load_data():
    df = pd.read_csv('qa_dataset_with_embeddings.csv')
    df['Question_Embedding'] = df['Question_Embedding'].apply(eval).apply(np.array)
    return df

df = load_data()

# Load the embedding model
@st.cache_resource
def load_model():
    return SentenceTransformer('paraphrase-MiniLM-L6-v2')

model = load_model()

# Function to find the most relevant answer
def find_answer(user_question, df, model, threshold=0.75):
    user_embedding = model.encode([user_question])
    similarities = cosine_similarity(user_embedding, list(df['Question_Embedding']))
    max_sim_idx = np.argmax(similarities)
    max_sim_score = similarities[0, max_sim_idx]
    
    if max_sim_score >= threshold:
        return df.iloc[max_sim_idx]['Answer'], max_sim_score
    else:
        return None, max_sim_score

# Streamlit interface
st.title("Smart FAQ Assistant for Health Topics")
st.write("Ask any question about heart, lung, and blood-related health topics.")

user_question = st.text_input("Enter your question:")
search_button = st.button("Find Answer")

if search_button and user_question:
    answer, score = find_answer(user_question, df, model)
    if answer:
        st.write(f"**Answer:** {answer}")
        st.write(f"**Similarity Score:** {score:.2f}")
    else:
        st.write("I apologize, but I don't have information on that topic yet. Could you please ask other questions?")

# Optional features
clear_button = st.button("Clear")
if clear_button:
    user_question = ""

# Display common FAQs
st.write("### Common FAQs:")
common_faqs = df.sample(5)  # Display 5 random FAQs
for i, row in common_faqs.iterrows():
    st.write(f"**Q:** {row['Question']}")
    st.write(f"**A:** {row['Answer']}")

