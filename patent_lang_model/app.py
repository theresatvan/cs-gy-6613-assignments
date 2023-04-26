import streamlit as st
from transformers import pipeline
from huggingface_hub import HfApi, ModelFilter

# Set up Hugging Face Hub API client
api = HfApi()

# Display title
st.title("Text Sentiment Analyzer")

# Retrieve all text classification models
models = api.list_models(filter=ModelFilter(task="text-classification"))[:10]
model_ids = [model.modelId for model in models]

# Create submission form
form = st.form("sentiment-form")
select_model = form.selectbox("Select a pretrained model", model_ids)
input = form.text_area('Enter your text here.')
submit = form.form_submit_button("Submit")

if submit:
    # Create pipeline to user's selected pre-trained model
    classifier = pipeline(task="sentiment-analysis", model=select_model)
    
    # Extract prediction from the results
    pred = classifier(input)
    
    # Display prediction
    st.write(pred)
