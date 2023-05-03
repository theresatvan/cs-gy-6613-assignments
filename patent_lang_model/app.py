import torch
import streamlit as st
from datasets import load_dataset
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer

decision_to_str = {'REJECTED': 0, 'ACCEPTED': 1, 'PENDING': 2, 'CONT-REJECTED': 3, 'CONT-ACCEPTED': 4, 'CONT-PENDING': 5}

dataset_dict = load_dataset('HUPD/hupd',
    name='sample',
    data_files="https://huggingface.co/datasets/HUPD/hupd/blob/main/hupd_metadata_2022-02-22.feather", 
    icpr_label=None,
    train_filing_start_date='2016-01-01',
    train_filing_end_date='2016-01-21',
    val_filing_start_date='2016-01-22',
    val_filing_end_date='2016-01-31',
)

dataset = dataset_dict['validation']

model_abstract = DistilBertForSequenceClassification.from_pretrained('theresatvan/hupd-distilbert-abstract')
tokenizer_abstract = DistilBertTokenizer.from_pretrained('theresatvan/hupd-distilbert-abstract')

model_claims = DistilBertForSequenceClassification.from_pretrained('theresatvan/hupd-distilbert-claims')
tokenizer_claims = DistilBertTokenizer.from_pretrained('theresatvan/hupd-distilbert-claims')


def predict(model_abstract, model_claims, tokenizer_abstract, tokenizer_claims, input):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    model_abstract.to(device)
    model_claims.to(device)
    
    model_abstract.eval()
    model_claims.eval()
    
    abstract, claims = input['abstract'], input['claims']
    
    encoding_abstract = tokenizer_abstract(abstract, return_tensors='pt', truncation=True, padding='max_length')
    encoding_claims = tokenizer_claims(claims, return_tensors='pt', truncation=True, padding='max_length')
    
    input_abstract = encoding_abstract['input_ids'].to(device)
    attention_mask_abstract = encoding_abstract['attention_mask'].to(device)
    input_claims = encoding_claims['input_ids'].to(device)
    attention_mask_claims = encoding_claims['attention_mask'].to(device)
    
    with torch.no_grad():
        outputs_abstract = model_abstract(input_ids=input_abstract)
        outputs_claims = model_claims(input_ids=input_claims)
        
    print(outputs_abstract.logits)
    print(outputs_claims.logits)
        
    combined_prob = (outputs_abstract.logits.softmax(dim=1) + outputs_claims.logits.softmax(dim=1)) / 2
    label = torch.argmax(combined_prob, dim=1)
    
    return label, combined_prob.tolist()[0]
    
    
if __name__ == '__main__':
    st.title = "Can I Patent This?"
    
    form = st.form('patent-prediction-form')
    dropdown = [example['patent_number'] for example in dataset]
    
    input_application = form.selectbox('Select a patent\'s application number', dropdown)
    submit = form.form_submit_button("Submit")
    
    if submit:
        input = dataset.filter(lambda e: e['patent_number'] == input_application)
        
        label, prob = predict(model_abstract, model_claims, tokenizer_abstract, tokenizer_claims, input)
        
        st.write(label)
        st.write(prob)
        st.write(input['decision'])
    