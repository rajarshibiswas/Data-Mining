# Final Project
#
# Author :  Rajarshi Biswas
#           Sayam Ganguly

import _pickle as pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import LinearSVC
import sys
from sklearn import metrics
from sklearn.metrics import classification_report

def data_label_split(data):
    X = data['TITLE']
    Y = data['CATEGORY']
    return X,Y

stdout = sys.stdout
training_set = pickle.load(open("training_set.p","rb"))
testing_set = pickle.load(open("testing_set.p","rb"))

train_x,train_y = data_label_split(training_set)
test_x,test_y = data_label_split(testing_set)

vectorizer = TfidfVectorizer()

vectorized_train = vectorizer.fit_transform(train_x)
vectorized_test = vectorizer.transform(test_x)

pickle.dump(vectorizer,open("TFIDF_LRG_Model.pk","wb"))

classifier_set = [(MultinomialNB(),'Multinomial Naive Bayes'),
                  (LogisticRegression(),'Logistic Regression'),
                  (LinearSVC(),'Linear SVM'),
                  (RandomForestClassifier(),'Random Forrest'),
                  (AdaBoostClassifier(),'AdaBoost')]

class_names = ['Business', 'Technology', 'Entertainment', 'Medicine']

pred_results = {}
i=0                  
for elem in classifier_set:
    model = elem[0]
    model_name = elem[1]
    print("Running",model_name)
    classifier = model.fit(vectorized_train, train_y)
    prediction = classifier.predict(vectorized_test)
    
    precision = metrics.precision_score(test_y, prediction,average=None)
    recall = metrics.recall_score(test_y, prediction,average=None)
    f1 = metrics.f1_score(test_y, prediction,average=None)
    accuracy = metrics.accuracy_score(test_y, prediction)
    confusion_matrix = metrics.confusion_matrix(test_y, prediction, labels=['b','t','e','m'])
    report = classification_report(test_y, prediction, target_names=class_names)
    
    d = {'precision':precision,
         'recall':recall,
         'F1':f1,
         'accuracy':accuracy,
         'confusion_matrix':confusion_matrix,
         'report':report}
    pred_results[model_name] = d
    if i==1:
        pickle.dump(model,open("TFIDF_LRG_Model.sav","wb"))
    print("Done......")
    i=i+1
    
sys.stdout = open("tfidf_results.txt", 'a')
    
for model_name,result in pred_results.items():
    print ('-------'+'-'*len(model_name))
    print ('MODEL:', model_name)
    print ('-------'+'-'*len(model_name))
    
    print ('Precision = ' + str(result['precision']))
    print ('Recall = ' + str(result['recall']))
    print ('F1 = ' + str(result['F1']))
    print ('Accuracy = ' + str(result['accuracy']))
    print ('Confusion matrix =  \n' + str(result['confusion_matrix']))
    print ('\nClassification Report:\n' + str(result['report']))

sys.stdout.close()
sys.stdout = stdout
print("Completed!")