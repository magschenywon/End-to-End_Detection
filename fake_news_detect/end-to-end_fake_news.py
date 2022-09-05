import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

data = pd.read_csv("news.csv")
#news.csv inludes: news title, news content and label (shows whether the news is fake/real)

##########################################################################################
#extract only-needed columns and split into training/test sets (no missing data from original file)
x = np.array(data["title"]) # use the 'title' column as the feature we need to train
y = np.array(data["label"]) # use the 'label' column as the values we want to predict

cv = CountVectorizer() 
x = cv.fit_transform(x)
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2, random_state=42)
model = MultinomialNB()
model.fit(xtrain, ytrain)

##########################################################################################
#this is the function to detect your input at webpage
import streamlit as st
st.title("Fake News Detection System")
def fakenewsdetection():
    user = st.text_area("Enter Any News Headline: ")
    if len(user) < 1:
        st.write("  ")
    else:
        sample = user
        data = cv.transform([sample]).toarray()
        a = model.predict(data)
        st.title(a)
fakenewsdetection()


# python3 -m venv venv 
# source venv/bin/activate 
# conda activate
# python3 -m pip install scikit-learn
# streamlit run end-to-end_fake_news.py
