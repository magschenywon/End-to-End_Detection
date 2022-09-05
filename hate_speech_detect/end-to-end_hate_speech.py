from nltk.util import pr
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

import nltk
nltk.download('stopwords')

data = pd.read_csv("twitter.csv") 
#print(data.head())
#this data is downloaded from kaggle and includes:
#columns: index, count, hate_speech, offensive_language, neighther, class, tweet


#add new columns to dataset: Hate Speech, Offensive Language, No Hate and Offensive
data["labels"] = data["class"].map({0: "Hate Speech", 1: "Offensive Language", 2: "No Hate and Offensive"})
#print(data.head())


#select only 'tweet' and 'labels' columns for the rest of test of training 
data = data[["tweet", "labels"]]
#print(data.head())

##########################################################################################
import re
import nltk
stemmer = nltk.SnowballStemmer("english")
from nltk.corpus import stopwords
import string
stopword=set(stopwords.words('english'))

#function 'clean' to clean up textx in 'tweet' column
def clean(text):
    text = str(text).lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    text = [word for word in text.split(' ') if word not in stopword]
    text=" ".join(text)
    text = [stemmer.stem(word) for word in text.split(' ')]
    text=" ".join(text)
    return text

data["tweet"] = data["tweet"].apply(clean) #apply clean function in 'tweet' col
#print(data.head())

##########################################################################################
#split dataset into training and test sets
x = np.array(data["tweet"])
y = np.array(data["labels"])

cv = CountVectorizer()
X = cv.fit_transform(x) # Fit the Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

clf = DecisionTreeClassifier()
clf.fit(X_train,y_train)
clf.score(X_test,y_test)

##########################################################################################
#this is the function to detect your input at webpage
def hate_speech_detection():
    import streamlit as st
    st.title("Hate Speech Detection")
    user = st.text_area("Enter any Tweet: ")
    if len(user) < 1:
        st.write("  ")
    else:
        sample = user
        data = cv.transform([sample]).toarray()
        a = clf.predict(data)
        st.title(a)
hate_speech_detection()

##########################################################################################
#To run this code:
#conda activate
#pip3 install streamlit
#streamlit run end-to-end_hate_speech.py

#using the streamlit library in Python which will help us see the predictions of the hate speech detection model in real-time
