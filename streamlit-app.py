import pandas as pd
import streamlit as st
import cleantext
import shap
from transformers import TextClassificationPipeline, AutoModelForSequenceClassification, AutoTokenizer

# Load the model and tokenizer
model_name = "ElKulako/cryptobert"
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)
pipe = TextClassificationPipeline(model=model, tokenizer=tokenizer, max_length=64, truncation=True, padding='max_length')

st.header('Sentiment Analysis')
with st.expander('Analyze Text'):
    text = st.text_input('Text here: ')
    if text:
        # Convert to list
        posts = text.tolist()
        explainer = shap.Explainer(pipe)
        shap_values = explainer(posts)
        shap_values
        shap.plots.text(shap_values[:, :, "Bullish"])
        # Plot the SHAP bar plot
        shap.plots.bar(shap_values[:, :, "Bearish"].mean(0))


    pre = st.text_input('Clean Text: ')
    if pre:
        st.write(cleantext.clean(pre, clean_all= False, extra_spaces=True ,
                                 stopwords=True ,lowercase=True ,numbers=True , punct=True))

with st.expander('Analyze CSV'):
    upl = st.file_uploader('Upload file')

    def score(x):
        blob1 = TextBlob(x)
        return blob1.sentiment.polarity

#
    def analyze(x):
        if x >= 0.5:
            return 'Positive'
        elif x <= -0.5:
            return 'Negative'
        else:
            return 'Neutral'

#
    if upl:
        df = pd.read_excel(upl)
        del df['Unnamed: 0']
        df['score'] = df['tweets'].apply(score)
        df['analysis'] = df['score'].apply(analyze)
        st.write(df.head(10))

        @st.cache
        def convert_df(df):
            # IMPORTANT: Cache the conversion to prevent computation on every rerun
            return df.to_csv().encode('utf-8')

        csv = convert_df(df)

        st.download_button(
            label="Download data as CSV",
            data=csv,
            file_name='sentiment.csv',
            mime='text/csv',
        )