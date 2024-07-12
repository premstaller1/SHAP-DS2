# Sentiment Analysis Using DistilRoberta, CryptoBERT, and SHAP

## Overview

This project aims to develop machine learning algorithms to conduct sentiment analysis using DistilRoberta and CryptoBERT models, along with SHAP for explainability. Additionally, the project explores the potential for price prediction in financial markets.

## Scope and Background

Recent technological advancements and significant events have greatly impacted financial markets, providing opportunities to forecast potential price changes. Understanding market characteristics such as uncertainty is crucial, particularly during periods of high volatility, such as the global Covid-19 pandemic. Studies have shown that news and social media play a crucial role in information exchange between experts and investors. The objective of this project is to develop a machine learning algorithm for sentiment analysis and implement SHAP for a comprehensive understanding of influencing factors and model explainability.

### State of the Art

The literature highlights the importance of model explainability for better understanding and improving model performance. Various studies have focused on implementing different models and identifying influential features:

1. **Goodell et al. (2023)**: Developed an explainable AI framework integrating SHAP to predict cryptocurrency prices, enhancing generalizability and interpretability.
   - Goodell, J. W., Jabeur, S. B., Saâdaoui, F., & Nasir, M. (2023). Explainable artificial intelligence modeling to forecast bitcoin prices. International Review Of Financial Analysis, 88, 102702. https://doi.org/10.1016/j.irfa.2023.102702

2. **Carbó and Gorjón (2024)**: Investigated machine learning methods to predict cryptocurrency volatility, finding that internal determinants like past volatility are most decisive.
   - Carbó, J. M., & Gorjón, S. (2024). Determinants of the price of bitcoin: An analysis with machine learning and interpretability techniques. International Review Of Economics & Finance, 92, 123–140. https://doi.org/10.1016/j.iref.2024.01.070

3. **Ueda et al. (2024)**: Proposed a social media text embedding method for predicting financial market volatility, confirming effective prediction with SHAP.
   - Ueda, K., Suwa, H., Yamada, M., Ogawa, Y., Umehara, E., Yamashita, T., Tsubouchi, K., & Yasumoto, K. (2024). SSCDV: Social media document embedding with sentiment and topics for financial market forecasting. Expert Systems With Applications, 245, 122988. https://doi.org/10.1016/j.eswa.2023.122988

4. **Adhikari et al. (2023)**: Enhanced sentiment analysis with an explainable hybrid word representation method, improving accuracy in financial news sentiment analysis.
   - Adhikari, S., Thapa, S., Naseem, U., Lu, H. Y., Bharathy, G., & Prasad, M. (2023). Explainable hybrid word representations for sentiment analysis of financial news. Neural Networks, 164, 115–123. https://doi.org/10.1016/j.neunet.2023.04.011

## Research Questions

1. What are the important features in the sentiment analysis of cryptocurrency and stock news/tweets using DistilRoberta-financial-sentiment/CryptoBERT and SHAP?
2. How do the predictions of sentiment analysis of cryptocurrency and stock news compare using SHAP?
3. How do the results of DistilRoberta-financial-sentiment (finetuned on financial news) compare with CryptoBERT (finetuned on crypto news)?
4. What are possible applications where explainable sentiment analysis could be used productively?

## Model Implementation and Dataset Processing

### Initial Setup

```python
import pandas as pd
import sklearn
import shap
import numpy as np
import torch
import seaborn as sns
import matplotlib.pyplot as plt
from transformers import TextClassificationPipeline, AutoModelForSequenceClassification, AutoTokenizer, pipeline
from datasets import load_dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
```

# Dataset Analysis

## Three datasets were used:

### DistilRoberta:
- **Source:** takala/financial_phrasebank from HuggingFace
- **Description:** Polar sentiment dataset of sentences from financial news.

### CryptoBERT:
- **Source:** ElKulako/stocktwits-crypto from HuggingFace
- **Description:** Contains cryptocurrency-related posts from the StockTwits website.

### Comparison Dataset:
- **Source:** StephanAkkerman/financial-tweets-stocks from HuggingFace
- **Description:** Ideal for training and evaluating models for sentiment analysis focused on market trends and investor sentiment.

# Data Preprocessing
- Balanced the CryptoBERT dataset to improve performance.
- Analyzed sentiment distributions and applied further processing.

# Model Accuracy Testing

### CryptoBERT:
```python
model_name = "ElKulako/cryptobert"
tokenizer_crypto = AutoTokenizer.from_pretrained(model_name, use_fast=True)
model_crypto = AutoModelForSequenceClassification.from_pretrained(model_name)
pipe_crypto = TextClassificationPipeline(model=model_crypto, tokenizer=tokenizer_crypto, max_length=64, truncation=True, padding='max_length')
```

## DistilRoberta:
```python
pipe_DR = pipeline("text-classification", model="mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis")
```

## Evaluation Metrics
Calculated accuracy, precision, recall, and F1-score for both models.

## SHAP Analysis

### Initial SHAP Analysis
Investigated the most important words influencing sentiment predictions for both models.

### Comparison on Same Dataset
Applied both models to the same dataset to compare SHAP results. Significant differences and similarities were observed in sentiment predictions.

## Web Application
A Streamlit web application was developed to provide interactive sentiment analysis using SHAP. The app can be accessed [here](https://sentiment-analysis-stock.streamlit.app/). The code and files are hosted on a separate GitHub repository: [SHAP_app](https://github.com/premstaller1/SHAP_app.git).

## Conclusion and Key Insights
The use of SHAP in sentiment analysis enhances understanding of model predictions by highlighting influential words. Data preprocessing and sentence structure are crucial for accurate sentiment analysis. DistilRoberta performed well on stock datasets, while CryptoBERT was effective for cryptocurrency-related tweets.

## Future Perspectives
Future projects could focus on integrating sentiment scores into models predicting stock or cryptocurrency prices, investigating the influence of specific features on predictions.
