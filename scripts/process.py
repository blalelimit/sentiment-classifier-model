import pandas as pd
import joblib

from sklearn.preprocessing import normalize
from sklearn.feature_extraction.text import TfidfVectorizer

from scripts.preprocess import *


# Get hashtags list
def get_hashtags():
    hashtags = pd.read_csv('datasets/hashtags_list.csv', sep=',', header=None)
    return flatten(hashtags.values)


# Get vocabulary
def get_vocabulary():
    vocabulary = pd.read_csv('datasets/feature_names.csv', sep=',', header=None)
    return flatten(vocabulary.values)


# Input entry/entries of strings
def input_sentiment():
    sentiment = input('Input your sentiment: ')
    return pd.DataFrame([sentiment], columns={'text'})


# Input entry/entries of strings
def input_model():
    model = input('Input the model (CNB/SVM/LR/RF/ENS): ')
    return model


# Load classifier model and predict
def predict_list(sentiment, model_name, cluster):
    if model_name not in ['CNB', 'SVM', 'LR', 'RF', 'ENS']:
        model_name = 'ENS'
    if cluster not in ['0', '1', '2', '3', '4', '5']:
        cluster = '0'

    model = joblib.load(f'models/{model_name}_Cluster{cluster}.pkl')
    prediction = model.predict(sentiment).tolist()[0]
    string = 'The ' + model_name + ' Model in Cluster ' + cluster + ' predicted '

    if prediction == -1:
        return string + 'NEGATIVE.'
    elif prediction == 0:
        return string + 'NEUTRAL.'
    elif prediction == 1:
        return string + 'POSITIVE.'
    else:
        return 'Invalid result'


# Load classifier model and predict on all clusters
def predict_all(sentiment, model_name):
    if model_name not in ['CNB', 'SVM', 'LR', 'RF', 'ENS']:
        model_name = 'ENS'
        print('ENS running by default')

    for x in range(6):
        model = joblib.load(f'models/{model_name}_Cluster{str(x)}.pkl')
        print(model_name + ' Cluster', str(x), 'predicted', model.predict(sentiment))


class Process:
    def __init__(self, sentiment):
        self.sentiment = sentiment

    # Preprocess entry
    def preprocess_list(self, hashtags):
        self.sentiment = PreProcess(self.sentiment, hashtags)
        self.sentiment.lower()
        self.sentiment.cleaning_a()
        self.sentiment.cleaning_b()
        self.sentiment.tokenization()
        return self.sentiment.lemmatization()

    # Tfidf Vectorizer
    def vectorize_list(self, vocabulary):
        df = self.sentiment.get_sentiment()
        if df['text_preprocessed'][0] == '[]':
            return list()
        else:
            tfidf = TfidfVectorizer(vocabulary=vocabulary)
            x = tfidf.fit_transform(df['text_preprocessed'])
            df = pd.DataFrame(normalize(x).toarray(), columns=tfidf.get_feature_names_out())
            if (df.T[0] == 0.0).all():
                return list()
            else:
                return df
