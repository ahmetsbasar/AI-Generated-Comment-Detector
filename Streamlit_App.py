import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib
import os

# Get the current directory of the script
current_dir = os.path.dirname(__file__)

# Construct the relative paths to the pkl files
tfidf_vectorizer_path = os.path.join(current_dir, 'tfidf_vectorizer.pkl')
logistic_regression_model_path = os.path.join(current_dir, 'logistic_regression_model.pkl')

# Load the TF-IDF vectorizer and logistic regression model
tfidf_vectorizer = joblib.load(tfidf_vectorizer_path)
logistic_regression_model = joblib.load(logistic_regression_model_path)

def preprocess_comment(comment):
    # Preprocess the input comment
    comment_tfidf = tfidf_vectorizer.transform([comment])
    return comment_tfidf

def predict_comment_authenticity(comment):
    # Preprocess the input comment
    comment_tfidf = preprocess_comment(comment)
    
    # Make prediction
    prediction = logistic_regression_model.predict(comment_tfidf)
    return prediction

def main():
    st.title("AI Generated Comment Detector")
    st.write("Enter a comment to check its authenticity:")

    comment = st.text_input("Enter comment here:")

    if st.button("Check Authenticity"):
        if comment.strip() == "":
            st.error("Please enter a comment.")
        else:
            prediction = predict_comment_authenticity(comment)
            if prediction == 1:
                st.success("The comment is likely to be AI-generated.")
            else:
                st.success("The comment is likely to be genuine.")

if __name__ == "__main__":
    main()
