# Sentiment Analysis Using DistilRoberta, CryptoBERT, and SHAP

## Overview

Github https://github.com/premstaller1/SHAP-DS2

The objective of this project is to implement, apply and compare different mashine learning models based on financial datasets. Additionally, the implementation of SHAP therefore allows to gain a comprehensive understanding of influencing factors and to explore the performance of the models.

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
- **Source:** takala/financial_phrasebank from HuggingFace (https://huggingface.co/datasets/takala/financial_phrasebank#data-fields)

### CryptoBERT:
- **Source:** ElKulako/stocktwits-crypto from HuggingFace (https://huggingface.co/datasets/ElKulako/stocktwits-crypto)

### Comparison Dataset:
- **Source:** StephanAkkerman/financial-tweets-stocks from HuggingFace (https://huggingface.co/datasets/StephanAkkerman/financial-tweets-stocks)

# Data Preprocessing
- Balanced the CryptoBERT dataset to improve performance.
- Analyzed sentiment distributions and applied further processing.

# Model Accuracy Testing

### CryptoBERT: (https://huggingface.co/ElKulako/cryptobert?text=I+hate+bitcoin)
```python
model_name = "ElKulako/cryptobert"
tokenizer_crypto = AutoTokenizer.from_pretrained(model_name, use_fast=True)
model_crypto = AutoModelForSequenceClassification.from_pretrained(model_name)
pipe_crypto = TextClassificationPipeline(model=model_crypto, tokenizer=tokenizer_crypto, max_length=64, truncation=True, padding='max_length')
```

## DistilRoberta: (https://huggingface.co/mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis)
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

## Conclusion
Overall, the use of SHAP in sentiment analyses for the distilroberta and cryptobert models leads to a better understanding in terms of transparency, explainability and accountability.

## Future Perspectives
Integration of sentiment scores into models predicting stock or cryptocurrency prices, and investigating the influence of specific features on predictions.
