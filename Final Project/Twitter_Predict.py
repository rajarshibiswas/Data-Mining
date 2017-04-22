# Final Project
#
# Author :  Rajarshi Biswas
#           Sayam Ganguly

import _pickle as pickle
import glob
import tweepy as tp
import pandas as pd
from tweepy import Stream
from tweepy.streaming import StreamListener
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import string
import sys


def tokenize(words):
    return word_tokenize(words)
    
    
def stem(token_list):
    ps = PorterStemmer()
    return [ps.stem(tok) for tok in token_list]
    

def remove_stopwords(words):
    stop_words = set(stopwords.words('english'))
    stop_words.add("new")
    stop_words.add("say")
    stop_words.add("may")
    stop_words.add("rt")
    words = [w for w in words if not w in stop_words]
    return words
    
    
def remove_punctuations(words):
    table = str.maketrans({key: None for key in string.punctuation})
    return words.translate(table)
    
    
def remove_digits(words):
    table = str.maketrans({key: None for key in string.digits})
    return words.translate(table)
    
    
def preprocess(text):
    text = text.lower()
    text = remove_punctuations(text)
    text = remove_digits(text)
    token_list = tokenize(text)
    token_list = stem(token_list)
    token_list = remove_stopwords(token_list)
    text = ' '.join(token_list)
    return text

class listener(StreamListener):
    def on_data(self,data):
        text = data.split(',"text":"')[1].split('","')[0]
        classify(text)
        return True
    def on_error(self,status):
        print("error")
        print(status)


def load_models():
    for file in glob.glob("*.sav"):
        model = pickle.load(open(file,'rb'))
        vectorizer = pickle.load(open(file[:-4]+'.pk','rb'))
        tup = (model,vectorizer)
        models.append(tup)
        

def connect_twitter():
    consumer_key = "AzmC1JpWz5MgLIzUXAgK7B6mE"
    consumer_secret = "TVbaKM8sOgHqh4gWvgtU2WshnOQC4SMIgzcIEk0NO2RgKrWzhy"
    access_token = '147602744-HS3qk582GtDvWAfYKRLkAgt3znXxSxbwWjBHuUPg'
    access_token_secret = 'F03uQ3m4kpRQllZrMHm8gYbS56OMDyrd8UP79USihcWFS'
    auth =tp.OAuthHandler(consumer_key = consumer_key, consumer_secret = consumer_secret)
    auth.set_access_token(access_token,access_token_secret)
    twitterStream = Stream(auth,listener())
    return twitterStream
    
def classify(text):
    global prediction_result ,twitterStream
    prediction = []
    print(text)
    prediction.append(text)
    text = preprocess(text)
    for model in models:
        l = []
        l.append(text)
        l = model[1].transform(l)
        x = model[0].predict(l)
        prediction.append(class_name[x[0]])
    d = {colNames[0]:prediction[0],colNames[1]:prediction[1],
         colNames[2]:prediction[2]}
    prediction_result = prediction_result.append(d,
                                                 ignore_index = True)
    if(prediction_result.shape[0] == 50):
        prediction_result.to_csv("Twitter_Prediction_Result.csv")
        sys.exit("Thanks for Using Twitter_Predict!!!")

        
models = []
load_models()
colNames = ["Twitter Text","Multinomial Naive Bayes",
            "Logistic Regression"]
prediction_result = pd.DataFrame(columns=colNames)
twitterStream = connect_twitter()
class_name = {'b':'Business','e':'Entertainment',
               't':'Technology','m':'Health'}
twitterStream.filter(track=["google", "apple", "microsoft", "facebook"],languages=['en'])



    