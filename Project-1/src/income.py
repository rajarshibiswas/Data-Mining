import numpy as np
import pandas as pd

def analyze_data():
    fileName = '../DataSet/income_tr.csv'
    # fileName = '..\DataSet\Iris.csv'
    data = read_data_file(fileName)
    prepare_data(data)

def read_data_file(fileName):
    try:
        # read the file
        data = pd.read_csv(fileName)
    except IOError:
        print ("Could not open file: "), fileName
    return data

def prepare_data(data):
    dict_edu = make_dictionary(data, 'education')
    dict_work_class = make_dictionary(data, 'workclass')
    dict_mar_status = make_dictionary(data, 'marital_status')
    dict_occupation = make_dictionary(data, 'occupation')
    dict_relationship = make_dictionary(data, 'relationship')
    dict_race = make_dictionary(data, 'race')
    dict_gender = make_dictionary(data, 'gender')
    dict_country = make_dictionary(data, 'native_country')
    print dict_country

def make_dictionary(data,colName):
    d = dict(zip(data[colName].unique(), np.arange(1, data[colName].unique().size)))
    return d

def convert_to_discrete(data,d):
    data1 = data['education'].where(data['education'] == d.items()[0][0]).dropna()

analyze_data()