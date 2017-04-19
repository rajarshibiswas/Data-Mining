# -*- coding: utf-8 -*-
"""
Created on Mon Apr 17 21:55:29 2017

@author: Sayam Ganguly
"""
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import string
import _pickle as pickle

def strip_data(data):
    data = data[['TITLE','CATEGORY']]
    return data


def tokenize(words):
    return word_tokenize(words)
    
    
def stem(words):
    ps = PorterStemmer()
    return ps.stem(words)
    

def remove_stopwords(words):
    stop_words = set(stopwords.words('english'))
    words = [w for w in words if not w in stop_words]
    return words
    
    
def remove_punctuations(words):
    table = str.maketrans({key: None for key in string.punctuation})
    return words.translate(table)
    
    
def remove_digits(words):
    table = str.maketrans({key: None for key in string.digits})
    return words.translate(table)
    
    
def preprocess(news_data):
    for index,row in news_data.iterrows():
        if index % 100 == 0:
            print("Completed - ",index)
        text = row[0].lower()
        text = remove_punctuations(text)
        text = remove_digits(text)
        token_list = tokenize(text)
        token_list = remove_stopwords(token_list)
        text = ' '.join(token_list)
        news_data.loc[index,'TITLE'] = text
    return news_data


def split_train_and_test(data):
    train_count = int(data.shape[0] * 0.7)
    train = data.head(train_count)
    test  = data.tail(1-train_count)
    return train,test

news_data = pd.read_csv("uci-news-aggregator.csv")
#Remove unused columns
news_data = strip_data(news_data)

#Preprocess data
news_data = preprocess(news_data)

#Separate Categories
data_b = news_data[news_data['CATEGORY']=='b']
data_t = news_data[news_data['CATEGORY']=='t']
data_e = news_data[news_data['CATEGORY']=='e']
data_m = news_data[news_data['CATEGORY']=='m']

#Split training and tesing data 70%-30% classwise
train_b,test_b  = split_train_and_test(data_b)
train_t,test_t  = split_train_and_test(data_t)
train_e,test_e  = split_train_and_test(data_e)
train_m,test_m  = split_train_and_test(data_m)

#Recreate training set
all_train = pd.DataFrame()
all_train = all_train.append(train_b)
all_train = all_train.append(train_t)
all_train = all_train.append(train_e)
all_train = all_train.append(train_m)

#Recreate testing set
all_test = pd.DataFrame()
all_test = all_test.append(test_b)
all_test = all_test.append(test_t)
all_test = all_test.append(test_e)
all_test = all_test.append(test_m)

#Save full Training Data
pickle.dump(all_train,open("training_set.p","wb"))
pickle.dump(all_test,open("testing_set.p","wb"))

#Save catgorized training data
pickle.dump(train_b,open("train_b.p","wb"))
pickle.dump(train_t,open("train_t.p","wb"))
pickle.dump(train_e,open("train_e.p","wb"))
pickle.dump(train_m,open("train_m.p","wb"))

#Save catgorized testing data
pickle.dump(test_b,open("test_b.p","wb"))
pickle.dump(test_t,open("test_t.p","wb"))
pickle.dump(test_e,open("test_e.p","wb"))
pickle.dump(test_m,open("test_m.p","wb"))


