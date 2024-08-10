import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import openai

openai.api_key = st.secrets["mykey"]

# Replace with your embedding model
model = "text-embedding-ada-002"

# Load your dataset
try:
    df = pd.read_csv('qa_dataset_with_embeddings.csv')
    # Convert the 'Question_Embedding' column from string to actual NumPy arrays
    df['Question_Embedding'] = df['Question_Embedding'].apply(lambda x: np.array(eval(x)))
except Exception as e:
    st.error(f"Error loading CSV file: {e}")

# Function to get embedding
def get_embedding(text):
    response = openai.Embedding.create(
        input=[text],
        model=model
    )
    embedding = response['data'][0]['embedding']
    return np.array(embedding).reshape(1, -1)

def find_best_answer(user_question):
    # Get embedding for the user's question
    user_question_embedding = get_embedding(user_question)

    # Calculate cosine similarities for all questions in the dataset
    df['Similarity'] = df['Question_Embedding'].apply(lambda x: cosine_similarity(x.reshape(1, -1), user_question_embedding).flatten()[0])

    # Find the most similar question and get its corresponding answer
    most_similar_index = df['Similarity'].idxmax()
    max_similarity = df['Similarity'].max()

    # Set a similarity threshold to determine if a question is relevant enough
    similarity_threshold = 0.6  # You can adjust this value

    if max_similarity >= similarity_threshold:
        best_answer = df.loc[most_similar_index, 'Answer']
        return best_answer, max_similarity
    else:
        return "I apologize, but I don't have information on that topic yet. Could you please ask other questions?", max_similarity

def main():
    st.title("Smart FAQ Assistant for Health Topics")
    st.write("Ask any question about heart, lung, and blood-related health topics.")

    # Search Bar for FAQs
    search_query = st.text_input("Search FAQs")
    if search_query:
        filtered_df = df[df['Question'].str.contains(search_query, case=False, na=False)]
        if not filtered_df.empty:
            st.subheader("Related FAQs")
            for _, row in filtered_df.iterrows():
                st.write(f"**Q:** {row['Question']}")
                st.write(f"**A:** {row['Answer']}")
        else:
            st.write("No FAQs found for the search query.")

    # Ask question section
    user_question = st.text_input("Ask your health question")
    if st.button("Submit"):
        if user_question:
            best_answer, similarity_score = find_best_answer(user_question)
            st.write(f"Similarity Score: {similarity_score:.2f}")
            st.write(best_answer)
            
            # Rating system
            rating = st.slider("Rate the helpfulness of the answer", 1, 5, 3)
            st.write(f"Thank you for rating this answer: {rating} star(s)")
        else:
            st.write("Please enter a question.")
    
    if st.button("Clear"):
        st.text_input("Ask your health question", value="", key="clear_input")
        st.write("")
        st.write("")

if __name__ == "__main__":
    main()
