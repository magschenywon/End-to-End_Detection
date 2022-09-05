import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
data = pd.read_csv("https://raw.githubusercontent.com/amankharwal/SMS-Spam-Detection/master/spam.csv", encoding= 'latin-1')

##########################################################################################
data = data[["class", "message"]]
x = np.array(data["message"]) # use the 'message' column as the feature we need to train
y = np.array(data["class"]) # use the 'class' column as the values we want to predict

cv = CountVectorizer()
X = cv.fit_transform(x) # Fit the Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

clf = MultinomialNB()
clf.fit(X_train,y_train)

##########################################################################################

sample = input('Enter a message:')
data = cv.transform([sample]).toarray()
print(clf.predict(data))

# (base) (venv) yiwenchen@Maggies-MacBook-Pro speam_detect% python3 test.py
# Enter a message:you just win a $20 gift card!
# ['spam']
