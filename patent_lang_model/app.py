import streamlit as st
from transformers import pipeline

st.title("Text Sentiment Analyzer")

# Create submission form
form = st.form("sentiment-form")
input = form.text_area('Enter your text here.')
submit = form.form_submit_button("Submit")

if submit:
    # Use pre-trained sentiment analysis model
    classifier = pipeline(task="sentiment-analysis")
    
    # Extract prediction from the results
    pred = classifier(input)[0]
    label= pred['label']
    score = pred['score']
    
    if label == "POSITIVE":
        # Green output text box
        st.success("{} sentiment (score: {})".format(label, score))
        
    else:
        # Red output text box
        st.error("{} sentiment (score: {})".format(label, score))
