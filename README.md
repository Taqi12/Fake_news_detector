# Fake News Detector

This is a Streamlit web application that detects fake news articles using a machine learning model.

## Overview

This project uses a Naive Bayes classifier trained on a dataset of real and fake news articles to predict the authenticity of user-entered news text.

## Files

* `streamlit_app.py`: The main Streamlit application code.
* `train_and_save_model.py`: Python script to train and save the machine learning model.
* `requirements.txt`: Lists the Python packages required to run the application.
* `tfidf_vectorizer.pkl`: The saved TF-IDF vectorizer model.
* `nb_classifier.pkl`: The saved Naive Bayes classifier model.
* `combined_news.csv`: The combined and preprocessed dataset.
* `True.csv`: Dataset containing real news articles.
* `Fake.csv`: Dataset containing fake news articles.

## How to Use

1.  Clone this repository to your local machine.
2.  Ensure you have Python and pip installed.
3.  Install the required packages: `pip install -r requirements.txt`
4.  Run the Streamlit app: `streamlit run streamlit_app.py`
5.  Enter a news article in the text area and click "Check" to see if it's real or fake.

## Model Training

The model is trained using the `train_and_save_model.py` script. The trained model and vectorizer are saved as `.pkl` files.

## Author

[Taqi Javed]
