#testing without streamlit

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


##########################################################################################
#test with sample
sample = "Let's unite and kill all the people who are protesting against the government"
data = cv.transform([sample]).toarray()
print(clf.predict(data))
##########################################################################################

#can't install sklearn in my machine. to run this code:
#python3 -m venv tutorial-env 
#source tutorial-env/bin/activate
#conda activate
#python3 test.py

#RESULT:
# (base) (tutorial-env) yiwenchen@Maggies-MacBook-Pro end-to-end_detection % python3 test.py
# [nltk_data] Downloading package stopwords to
# [nltk_data]     /Users/yiwenchen/nltk_data...
# [nltk_data]   Package stopwords is already up-to-date!
# ['Hate Speech']
