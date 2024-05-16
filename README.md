# Financial Markets Prediction Using Social-Media Sentiment Analysis

## Project Overview

### Title
Financial Markets Prediction Using Social-Media Sentiment Analysis (e.g., Reddit/Twitter/News)

### Background
In recent years, technological advancements and significant global events have had a substantial impact on financial markets. This has provided investors, traders, and researchers with opportunities to develop various methods for forecasting potential price changes. Understanding the characteristics of financial markets, such as uncertainty and market sentiment, is crucial. The global Covid-19 pandemic, for example, highlighted increased market volatility and saw Bitcoin reach all-time highs in 2021 and 2024. 

Studies have identified social media and news platforms as critical venues for the exchange of information and opinions among experts and investors. Consequently, the goal of this project is to develop a machine learning algorithm that conducts sentiment analysis. By implementing SHAP (Shapley Additive exPlanations), we aim to gain a comprehensive understanding of the influencing factors and explore various models for predicting cryptocurrency prices.

### State of the Art
#### Goodell et al. (2023)
Goodell et al. (2023) explore the use of an explainable artificial intelligence (XAI) framework to predict cryptocurrency behavior. They highlight that many existing methods lack explanatory power. Their study integrates XAI with a SHAP-based approach to improve the generalizability and interpretability of forecasting models. The results demonstrated that XAI modeling could effectively identify critical factors influencing cryptocurrency prices during significant events such as the Russian-Ukraine war.

*Goodell, J. W., Jabeur, S. B., Saâdaoui, F., & Nasir, M. (2023). Explainable artificial intelligence modeling to forecast bitcoin prices. International Review Of Financial Analysis, 88, 102702. https://doi.org/10.1016/j.irfa.2023.102702*

#### Carbó and Gorjón (2024)
Carbó and Gorjón (2024) examine different machine learning methods to predict cryptocurrency volatility. Their study confirms that random forest and long-short term memory (LSTM) networks perform better on complex cryptocurrencies than previous models. By integrating SHAP, they determined that internal determinants, such as past volatility and trading data, were the most significant. Despite some limitations, their study contributes to a deeper understanding of the cryptocurrency market.

*Carbó, J. M., & Gorjón, S. (2024). Determinants of the price of bitcoin: An analysis with machine learning and interpretability techniques. International Review Of Economics & Finance, 92, 123–140. https://doi.org/10.1016/j.iref.2024.01.070*

#### Ueda et al. (2024)
Ueda et al. (2024) propose a social media text embedding method to predict financial market volatility. Their novel document embedding technique focuses on the relationship between topic and sentiment. Their model demonstrates high prediction performance, especially during periods of increased market volatility. The study's findings contribute to reducing investment risks by improving prediction accuracy.

*Ueda, K., Suwa, H., Yamada, M., Ogawa, Y., Umehara, E., Yamashita, T., Tsubouchi, K., & Yasumoto, K. (2024). SSCDV: Social media document embedding with sentiment and topics for financial market forecasting. Expert Systems With Applications, 245, 122988. https://doi.org/10.1016/j.eswa.2023.122988*

### Research Questions
1. What are the important features in the sentiment analysis of cryptocurrency and stock news using DistilRoberta-financial-sentiment and SHAP?
2. How do the predictions of a sentiment analysis of cryptocurrency and stock news compare using SHAP?
3. How do the results of DistilRoberta-financial-sentiment (fine-tuned on financial news) compare with cryptobert (fine-tuned on crypto news)?

## Model
- [DistilRoberta-financial-sentiment-analysis](https://huggingface.co/mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis)
- [CryptoBERT](https://huggingface.co/ElKulako/cryptobert?text=I+hate+bitcoin)

## Datasets
- [Crypto News Articles with Price Momentum Labels](https://huggingface.co/datasets/SahandNZ/cryptonews-articles-with-price-momentum-labels/viewer/default/train?p=1442)
- [Financial Tweets Crypto](https://huggingface.co/datasets/StephanAkkerman/financial-tweets-crypto/viewer/default/train?p=475)
- [StockTwits Crypto](https://huggingface.co/datasets/ElKulako/stocktwits-crypto)

## Getting Started

### Prerequisites
- Python 3.8 or higher
- Necessary Python packages (see `requirements.txt`)

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/financial-markets-prediction.git
   cd financial-markets-prediction
   ```
2. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

### Running the Project
1. Prepare the datasets by downloading and placing them in the `data/` directory.
2. Train the models:
   ```bash
   python train_model.py
   ```
3. Evaluate the models:
   ```bash
   python evaluate_model.py
   ```

### Usage
- For sentiment analysis:
   ```python
   from sentiment_analysis import analyze_sentiment
   sentiment = analyze_sentiment("Your text here")
   ```
- For SHAP analysis:
   ```python
   from shap_analysis import explain_model
   explanation = explain_model(model, data)
   ```

## Acknowledgements
We would like to thank the authors of the cited studies for their foundational work in this field. Their research has significantly contributed to our project.
