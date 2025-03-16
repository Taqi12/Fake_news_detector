import streamlit as st
import joblib
import pandas as pd
from sklearn.metrics import classification_report

def predict_fake_news(text):
    """Predicts if a news article is fake or real."""
    try:
        tfidf_vectorizer = joblib.load('tfidf_vectorizer.pkl')
        nb_classifier = joblib.load('nb_classifier.pkl')
        vectorized_text = tfidf_vectorizer.transform([text])
        prediction = nb_classifier.predict(vectorized_text)[0]
        return prediction
    except FileNotFoundError:
        return "Model files not found. Please train the model first."

def get_model_accuracy():
    """Calculates and returns the model accuracy for real news."""
    try:
        df = pd.read_csv("combined_news.csv")
        df = df.dropna()
        X = df['text']
        y = df['label']

        tfidf_vectorizer = joblib.load('tfidf_vectorizer.pkl')
        nb_classifier = joblib.load('nb_classifier.pkl')

        vectorized_text = tfidf_vectorizer.transform(X)
        y_pred = nb_classifier.predict(vectorized_text)

        report = classification_report(y, y_pred, output_dict=True)
        real_accuracy = report['True']['precision']
        return real_accuracy
    except FileNotFoundError:
        return None

# Streamlit app
st.set_page_config(page_title="Fake News Detector 📰", page_icon="📰", layout="wide")

st.title("Fake News Detector 📰🔍") # Added magnifying glass emoji
st.markdown("Enter a news article to check its authenticity. 🧐") # Added thinking face emoji

col1, col2, col3 = st.columns([1, 3, 1])

with col2:
    news_article = st.text_area("News Article:", height=200)

col_button1, col_button2, col_button3 = st.columns([1, 3, 1])

with col_button2:
    if st.button("Check 🕵️‍♂️", key="check_button"): # Added detective emoji
        if news_article:
            prediction = predict_fake_news(news_article)
            if prediction == "True":
                st.success("✅ This article is likely REAL. 👍") # Added thumbs up emoji
            else:
                st.error("❌ This article is likely FAKE. 👎") # Added thumbs down emoji
        else:
            st.warning("⚠️ Please enter a news article. 📝") # Added pencil emoji

real_acc = get_model_accuracy()
if real_acc is not None:
    st.markdown(f"**Model Accuracy 📊:**") # Added bar chart emoji
    st.markdown(f"- Real News Accuracy: {real_acc:.2%}")

st.markdown("""
<style>
[data-testid="stTextArea"] {
    border: 2px solid #4CAF50;
    border-radius: 5px;
    padding: 10px;
}
#check_button {
    background-color: #4CAF50 !important;
    color: white !important;
    padding: 10px 20px !important;
    border: none !important;
    border-radius: 5px !important;
    cursor: pointer !important;
    visibility: visible !important;
    display: inline-block !important;
}
body {
    color: #333;
    font-family: sans-serif;
}
.stSuccess {
    background-color: #d4edda;
    color: #155724;
    padding: 10px;
    border-radius: 5px;
}
.stError {
    background-color: #f8d7da;
    color: #721c24;
    padding: 10px;
    border-radius: 5px;
}
.stWarning {
    background-color: #fff3cd;
    color: #856404;
    padding: 10px;
    border-radius: 5px;
}
</style>
""", unsafe_allow_html=True)
