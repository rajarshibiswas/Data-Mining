import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas import DataFrame
import warnings


def analyze_data():
    warnings.filterwarnings("ignore")
    fileName = '../DataSet/income_tr.csv'
    # fileName = '..\DataSet\Iris.csv'
    data = read_data_file(fileName)
    data = prepare_data(data)
    #euclidean_distance(data[:10], 5)
    #cosine_similarity(data[:10], 5)
    print data.loc[1]

    print data.loc[4]
    #plot_data(data)


def read_data_file(fileName):
    try:
        # read the file
        data = pd.read_csv(fileName)
    except IOError:
        print ("Could not open file: "), fileName
    return data


def remove_missing_values(data):
    return data.loc[data['workclass'] != ' ?']


def remove_ambiguous_columns(data):
    data.drop('ID', axis=1, inplace=True)
    data.drop('fnlwgt', axis=1, inplace=True)
    data.drop('relationship', axis=1, inplace=True)
    data.drop('class', axis=1, inplace=True)


def categorize_age(data):
    bins = [0, 30, 50, 100]
    group_names = [1, 2, 3]
    data['age'] = pd.cut(data['age'], bins, labels=group_names)


def categorize_education(data):
    bins = [0, 8, 10, 12, 13, 14, 16]
    group_names = [1, 2, 3, 4, 5, 6]
    data['education'] = pd.cut(data['education_cat'], bins, labels=group_names)
    data.drop('education_cat', axis=1, inplace=True)


def categorize_workclass(data):
    data.loc[data['workclass'].str.contains("Private"), 'workclass_cat'] = 1
    data.loc[data['workclass'].str.contains("Self"), 'workclass_cat'] = 2
    data.loc[data['workclass'].str.contains("gov"), 'workclass_cat'] = 3
    data['workclass'] = data['workclass_cat']
    data.drop('workclass_cat', axis=1, inplace=True)
    data.workclass = data.workclass.astype(int)


def categorize_occupation(data):
    data.occupation = data.occupation.astype('category')
    data.occupation = data.occupation.cat.codes + 1


def categorize_marital_status(data):
    data.loc[data['marital_status'].str.contains(" Married"), 'marital_status_cat'] = 1
    data.loc[[not d for d in data['marital_status'].str.contains(" Married")], 'marital_status_cat'] = 2
    data['marital_status'] = data['marital_status_cat']
    data.drop('marital_status_cat', axis=1, inplace=True)


def categorize_country(data):
    data.loc[data['native_country'].str.contains(" United-States"), 'native_country_cat'] = 1
    data.loc[[not d for d in data['native_country'].str.contains(" United-States")], 'native_country_cat'] = 2
    data['native_country'] = data['native_country_cat']
    data.drop('native_country_cat', axis=1, inplace=True)


def categorize_race(data):
    data.loc[data['race'].str.contains(" White"), 'race_cat'] = 1
    data.loc[[not d for d in data['race'].str.contains(" White")], 'race_cat'] = 2
    data['race'] = data['race_cat']
    data.drop('race_cat', axis=1, inplace=True)


def categorize_gender(data):
    data.gender = data.gender.astype('category')
    data.gender = data.gender.cat.codes + 1


def categorize_capital(data):
    data.loc[data['capital_gain']>0, 'capital_gain'] = 1
    data.loc[data['capital_loss']>0, 'capital_loss'] = 1


def categorize_hours(data):
    data.loc[data['hour_per_week'] < 40, 'hour_per_week'] = 1
    data.loc[data['hour_per_week'] == 40, 'hour_per_week'] = 2
    data.loc[data['hour_per_week'] > 40, 'hour_per_week'] = 3


# Compute euclidean distance
def euclidean_distance(data, k):
    # Get the number of rows in data
    num_data_frame_row = data.shape[0]

    # The result array
    eculidean_dis = []
    for i in range(num_data_frame_row):
        x = data.iloc[i].values[0:11]
        temp = []
        for j in range(num_data_frame_row):
            y = data.iloc[j].values[0:11]
            temp.append((np.sqrt(np.sum((x - y) ** 2)), j))
        eculidean_dis.append(temp)
    prepare_output(eculidean_dis, k, '../DataSet/Euclidean_Income.csv', False)


def prepare_output(distance_matrix, k, filename, similarity_flag):
    num_data_frame_row = len(distance_matrix)
    colnames = []
    for i in range(k - 1):
        colnames.append(str(i + 1))
        colnames.append(str(i + 1) + '-Prox')
    df = DataFrame(columns=colnames)
    # print the result
    for i in range(num_data_frame_row):
        # sort the tuple based on the euclidean_distance
        if similarity_flag:
            result = np.array(sorted(distance_matrix[i], key=lambda x: x[0], reverse=True))
        else:
            result = np.array(sorted(distance_matrix[i], key=lambda x: x[0]))
        l = []
        for j in range(k - 1):
            l.append(result[j + 1][1]+1)
            l.append(result[j + 1][0])
        df.loc[i] = l
    df.columns.name = "Transaction ID"
    df.index += 1
    df.to_csv(path_or_buf=filename)


# Cosine Similarity proximity function.
def cosine_similarity(df, k):
    dataRows = df.shape[0]
    dotMatrix = df.dot(df.T)
    cos = []
    for i in range(dataRows):
        temp = []
        for j in range(dataRows):
            if i != j and j in df.index and i in df.index:
                # cos[i][j] = dotMatrix[i][j]/(np.sqrt(dotMatrix[i][i]) * np.sqrt(dotMatrix[j][j]))
                val = dotMatrix[i][j] / (np.sqrt(dotMatrix[i][i]) * np.sqrt(dotMatrix[j][j]))
                temp.append((val, j))
            else:
                temp.append((0.0, j))
        cos.append(temp)
    prepare_output(cos, k, '../DataSet/Cosine_Income.csv', True)



def prepare_data(data):
    # Remove Missing Value
    data = remove_missing_values(data)
    # Remove ID and fnlwgt columns
    remove_ambiguous_columns(data)
    # Generate Age groups
    categorize_age(data)
    # Categorize Education
    categorize_education(data)
    # Categorize Workclass
    categorize_workclass(data)
    # Categorize Occupation
    categorize_occupation(data)
    # Categorize Marital Status
    categorize_marital_status(data)
    # Categorize Country
    categorize_country(data)
    # Categorize race
    categorize_race(data)
    # Categorize Gender
    categorize_gender(data)
    # Categorize Capital Gain/Capital Loss
    categorize_capital(data)
    # Categorize Hours per Week
    categorize_hours(data)

    return data

"""
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
"""

def plot_data(data):
    plt.figure()
    #plt.scatter(data['education'],data['class'])
    data.occupation.value_counts().plot(kind='bar')
    #df = data.loc[:, ['education', 'class']].copy()
    #data['education'].hist()
    plt.show()


analyze_data()
