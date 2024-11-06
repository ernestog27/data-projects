# Building a RESTful API for Sentiment Analysis with Flask
# Ernesto Gonzales, MSDA

import pandas as pd

#Loading the dataset
data = pd.read_csv('sentiment_env/databases/sentiment labelled sentences/imdb_labelled.txt', delimiter = '\t', header = None)
data.columns = ['Sentence', 'Label'] # Rename columns for clarity

# Data preview
print(data.head())
print(data.info())
print(data['Label'].value_counts())

# Data Cleaning and Pre-processing 

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re

# Downloading necessary NLTK data

nltk.download('stopwords')
nltk.download('punkt_tab')

# Function for text cleaning

def preprocess_text(text):
    text = re.sub(r'\W', ' ', text) # Remove non-word characters
    text = text.lower() # Convert text to lowercase
    words = word_tokenize(text) # Tokenize text
    words = [word for word in words if word not in stopwords.words('english')] # Remove stopwords
    return ' '.join(words)

# Applying function to the text column

data['Cleaned_Sentence'] = data['Sentence'].apply(preprocess_text)

# Spliting data into training and testing sets

from sklearn.model_selection import train_test_split
X = data['Cleaned_Sentence']
y = data['Label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

# Feature extraction

from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(max_features = 5000)
X_train_tdif = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Training and Evaluating model

from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
model.fit(X_train_tdif, y_train)

from sklearn.metrics import accuracy_score, classification_report

y_pred = model.predict(X_test_tfidf)
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
print(classification_report(y_test,y_pred))

from sklearn.metrics import ConfusionMatrixDisplay

ConfusionMatrixDisplay.from_predictions(y_test, y_pred)

# Preparation for Deployment

import joblib

joblib.dump(model, 'model.pkl')
joblib.dump(vectorizer, 'vectorizer.pkl')

# Creating a Simple API with Flask

from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

# Loading the saved model and vectorizer

model = joblib.load('model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

@app.route('/predict', methods = ['POST'])
def predict():
    data = request.get_json(force = True)
    sentence = data['sentence']
    sentence_tfidf = vectorizer.transform([sentence])
    prediction = model.predict(sentence_tfidf)
    return jsonify({'prediction': int(prediction[0])})

if __name__ == '__main__':
    app.run(debug=True)
    
