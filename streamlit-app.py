import pandas as pd
import streamlit as st
import cleantext
import shap
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Load the model and tokenizer
model_name = "ElKulako/cryptobert"
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)

# Function to preprocess text and get SHAP values
def preprocess_and_shap(text):
    # Preprocess text
    cleaned_text = cleantext.clean(text, clean_all=False, extra_spaces=True, stopwords=True,
                                   lowercase=True, numbers=True, punct=True)
    # Tokenize text
    inputs = tokenizer(cleaned_text, return_tensors='pt', max_length=64, truncation=True, padding='max_length')
    # Compute SHAP values
    explainer = shap.Explainer(model, tokenizer)
    shap_values = explainer(inputs)
    return cleaned_text, shap_values

# Function to predict sentiment
def predict_sentiment(text):
    inputs = tokenizer(text, return_tensors='pt', max_length=64, truncation=True, padding='max_length')
    outputs = model(**inputs)
    sentiment_probabilities = outputs.logits.softmax(dim=1).detach().numpy()[0]
    sentiment_label = ['Negative', 'Neutral', 'Positive'][sentiment_probabilities.argmax()]
    return sentiment_label

st.header('Sentiment Analysis and SHAP Explanation')
with st.expander('Analyze Text'):
    text = st.text_input('Text here: ')
    if text:
        cleaned_text, shap_values = preprocess_and_shap(text)
        st.write('Cleaned Text:', cleaned_text)
        sentiment = predict_sentiment(text)
        st.write('Sentiment:', sentiment)
        # Visualize SHAP values
        st.write('SHAP Explanation:')
        shap.summary_plot(shap_values, feature_names=tokenizer.convert_ids_to_tokens(inputs['input_ids'][0]))

with st.expander('Analyze CSV'):
    upl = st.file_uploader('Upload file')

    if upl:
        df = pd.read_excel(upl)
        del df['Unnamed: 0']
        # Apply preprocessing and SHAP analysis to each tweet
        df['cleaned_text'], df['shap_values'] = zip(*df['tweets'].apply(preprocess_and_shap))
        # Predict sentiment for each tweet
        df['sentiment'] = df['cleaned_text'].apply(predict_sentiment)
        st.write(df.head(10))

        @st.cache
        def convert_df(df):
            # IMPORTANT: Cache the conversion to prevent computation on every rerun
            return df.to_csv().encode('utf-8')

        csv = convert_df(df)

        st.download_button(
            label="Download data as CSV",
            data=csv,
            file_name='sentiment_and_shap.csv',
            mime='text/csv',
        )