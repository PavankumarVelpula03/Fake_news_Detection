import streamlit as st
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from PIL import Image
from time import sleep

# Load the trained models and vectorizer
LR = joblib.load("logistic_regression.pkl")
DT = joblib.load("decision_tree.pkl")
GB = joblib.load("gradient_boosting.pkl")
RF = joblib.load("random_forest.pkl")
vectorization = joblib.load("tfidf_vectorizer.pkl")

# Function to classify news
def predict_fake_news(news):
    new_x_test = [news]
    new_xv_test = vectorization.transform(new_x_test)
    
    pred_LR = LR.predict(new_xv_test)[0]
    pred_DT = DT.predict(new_xv_test)[0]
    pred_GB = GB.predict(new_xv_test)[0]
    pred_RF = RF.predict(new_xv_test)[0]
    
    predictions = [pred_LR, pred_DT, pred_GB, pred_RF]
    majority_vote = sum(predictions) > len(predictions) / 2  # Majority Voting
    
    return "Not Fake News" if majority_vote else "Fake News"

# Streamlit UI
st.set_page_config(page_title="Fake News Detector", page_icon="📰", layout="wide")
st.markdown("""
    <style>
    .main {
        background-color: #f0f2f6;
    }
    .stTextArea textarea {
        font-size: 18px;
    }
    </style>
""", unsafe_allow_html=True)

st.title("📰 Fake News Detector")
st.write("Detect whether a news article is **Fake or Real** using Machine Learning.")

# Sidebar with instructions
st.sidebar.title("ℹ️ Instructions")
st.sidebar.write("Enter a news article text and click 'Check News' to verify its authenticity.")
st.sidebar.write("The model will analyze the input using multiple machine learning techniques.")
st.sidebar.write("Results are based on a majority voting system from multiple classifiers.")

# Adding an image with reduced size
image = Image.open("fake fact.jpg")
image = image.resize((600, 300))  # Resize image to reduce size
st.image(image, caption="Fake News Detection", use_column_width=False)

# User Input
news_input = st.text_area("📝 Enter News Text:", height=100)

if st.button("🔍 Check News"):
    if news_input.strip():
        with st.spinner("Analyzing the news...🔄"):
            sleep(2)  # Simulating processing time
            result = predict_fake_news(news_input)
        
        if result == "Fake News":
            st.error("🚨 **Fake News Detected!**", icon="🚫")
            fake_image = Image.open("fake.jpg")  # Ensure this image exists
            st.image(fake_image, caption="🚫 Fake News Detected", use_column_width=False)
        else:
            st.success("✅ **This seems to be Real News!**", icon="✔️")
            real_image = Image.open("fact.jpg")  # Ensure this image exists
            st.image(real_image, caption="✔️ Real News Confirmed", use_column_width=False)
    else:
        st.warning("⚠️ Please enter some text.")

# Footer
st.sidebar.markdown("""
---
**Developed by: Team 7855**

📧 Contact: [pavankumarvelpula02@gmail.com](mailto:pavankumarvelpula02@gmail.com)
""")
