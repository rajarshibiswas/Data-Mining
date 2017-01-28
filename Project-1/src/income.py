import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas import DataFrame

def analyze_data():
    fileName = '../DataSet/income_tr.csv'
    # fileName = '..\DataSet\Iris.csv'
    data = read_data_file(fileName)
    #prepare_data(data)
    #print data.head()
    plot_data(data)

def read_data_file(fileName):
    try:
        # read the file
        data = pd.read_csv(fileName)
    except IOError:
        print ("Could not open file: "), fileName
    return data

def prepare_data(data):
    dict_edu, dict_work_class, dict_mar_status, dict_occupation, \
    dict_relationship, dict_race, dict_gender, dict_country = prepare_dict(data)
    convert_to_class(data, 'education', dict_edu)
    convert_to_class(data, 'workclass', dict_work_class)
    convert_to_class(data, 'marital_status', dict_mar_status)
    convert_to_class(data, 'occupation', dict_occupation)
    convert_to_class(data, 'relationship', dict_relationship)
    convert_to_class(data, 'race', dict_race)
    convert_to_class(data, 'gender', dict_gender)
    convert_to_class(data, 'native_country', dict_country)


def prepare_dict(data):
    dict_edu = make_dictionary(data, 'education')
    dict_work_class = make_dictionary(data, 'workclass')
    dict_mar_status = make_dictionary(data, 'marital_status')
    dict_occupation = make_dictionary(data, 'occupation')
    dict_relationship = make_dictionary(data, 'relationship')
    dict_race = make_dictionary(data, 'race')
    dict_gender = make_dictionary(data, 'gender')
    dict_country = make_dictionary(data, 'native_country')
    return dict_edu, dict_work_class, dict_mar_status,dict_occupation,\
           dict_relationship,dict_race,dict_gender,dict_country

def make_dictionary(data,colName):
    d = dict(zip(data[colName].unique(), np.arange(0, data[colName].unique().size)))
    return d

def convert_to_class(data,colName,d):
    #data = data['education'].where(data['education'] == d.items()[0][0]).dropna()
    for item in d.items():
        data.loc[data[colName] == item[0], colName] = item[1]

def plot_data(data):
    plt.figure()
    #plt.scatter(data['education'],data['class'])
    data.occupation.value_counts().plot(kind='bar')
    #df = data.loc[:, ['education', 'class']].copy()
    #data['education'].hist()
    plt.show()


analyze_data()