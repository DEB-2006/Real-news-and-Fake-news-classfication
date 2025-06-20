import pandas as pd
import streamlit as st
import numpy as np
from matplotlib import pyplot as pylt
import seaborn as sns
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier , GradientBoostingClassifier
from sklearn.model_selection import train_test_split , GridSearchCV
from sklearn.metrics import accuracy_score , ConfusionMatrixDisplay , classification_report , roc_curve

#Load Data
news_df = pd.read_csv('WELFake_Dataset.csv', encoding='utf-8')
news_df = news_df.fillna(' ')
news_df['content'] = news_df['title'] + " " + news_df['text']
x = news_df.drop('label',axis=1)
y = news_df['label']

#Define stemming
ps = PorterStemmer
def stemming(content):
    stemmed_content = re.sub('[^a-zA-Z]'," ",content)
    stemmed_content = stemmed_content.lower()
    stemmed_content = stemmed_content.split()
    stemmed_content = [ps.stem(word) for word in stemmed_content if not word in stopwords.words('english')]
    stemmed_content = " ".join(stemmed_content)
    return stemmed_content

#Vetorize data

x = news_df['text'].values
y = news_df['label'].values
vector = TfidfVectorizer()
vector.fit(x)
x = vector.transform(x)

#Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2,stratify=y, random_state=1)

#Fit logistic regression model
model = LogisticRegression()
model.fit(X_train,y_train)

# Website
st.title('Fake news Detector')
input_text = st.text_input('Enter news Article')

def prediction(input_text):
    input_data = vector.transfrom([input_text])
    model.predict(input_data)
    return prediction[0]
if input_text:
    pred = prediction(input_text)
    if pred == 1:
        st.write('The news is fake.')
    else:
        st.write('The news is Real.')
