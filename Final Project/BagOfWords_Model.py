# -*- coding: utf-8 -*-
"""
Created on Wed Apr 19 12:10:04 2017

@author: Sayam Ganguly
"""

import _pickle as pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import sys
from sklearn import metrics
from sklearn.metrics import classification_report

def data_label_split(data):
    X = data['TITLE']
    Y = data['CATEGORY']
    return X,Y

training_set = pickle.load(open("training_set.p","rb"))
testing_set = pickle.load(open("testing_set.p","rb"))

train_x,train_y = data_label_split(training_set)
test_x,test_y = data_label_split(testing_set)

vectorizer = CountVectorizer(analyzer = "word",
                             tokenizer = None,    
                             preprocessor = None,
                             ngram_range = (1, 1),
                             binary = False,
                             strip_accents='unicode')

vectorized_train = vectorizer.fit_transform(train_x)
vectorized_test = vectorizer.transform(test_x)

classifier = MultinomialNB().fit(vectorized_train, train_y)
prediction = classifier.predict(vectorized_test)

model_name = 'Multinomial Naive Bayes'
class_names = ['Business', 'Technology', 'Entertainment', 'Medicine']

sys.stdout = open("bagofwords_results.txt", 'a')
    
print ('-------'+'-'*len(model_name))
print ('MODEL:', model_name)
print ('-------'+'-'*len(model_name))

print ('Precision = ' + str(metrics.precision_score(test_y, prediction,average=None)))
print ('Recall = ' + str(metrics.recall_score(test_y, prediction,average=None)))
print ('F1 = ' + str(metrics.f1_score(test_y, prediction,average=None)))
print ('Accuracy = ' + str(metrics.accuracy_score(test_y, prediction)))
print ('Confusion matrix =  \n' + str(metrics.confusion_matrix(test_y, prediction, labels=['b','t','e','m'])))
print ('\nClassification Report:\n' + classification_report(test_y, prediction, target_names=class_names))