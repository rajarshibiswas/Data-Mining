# Project 1
#
# Author :  Rajarshi Biswas
#           Sayam Ganguly

import pandas as pd
from pandas import DataFrame
import numpy as np
import matplotlib.pyplot as plt
np.set_printoptions(precision=3)


def plot_hist(data):
    df = data.loc[:, ['sepal_length', 'sepal_width', ' petal_length',' petal_width']].copy()
    plt.figure()
    print type(df)
    df[' petal_width'].hist()
    df.hist()
    plt.xlabel('Petal Width')
    plt.show()
    return


# Cosine Similarity proximity function.
def cosine_similarity(df, k):
    df = df.loc[:,['sepal_length','sepal_width',' petal_length',
    ' petal_width']]
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


def minkowski_distance(data, k, r):

    num_data_frame_row = data.shape[0]

    # The result array
    eculidean_dis = []

    for i in range(num_data_frame_row):
        x = data.iloc[i].values[0:4]
        temp = []
        for j in range(num_data_frame_row):
            y = data.iloc[j].values[0:4]
            temp.append((np.power(np.sum(np.power(np.absolute(x - y), r)), (1.0/r)), j))
        eculidean_dis.append(temp)

    prepare_output(eculidean_dis, k, '../DataSet/Minkowski.csv')


# Compute euclidean distance
def euclidean_distance(data, k):
    # Get the number of rows in data
    num_data_frame_row = data.shape[0]

    # The result array
    eculidean_dis = []
    for i in range(num_data_frame_row):
        x = data.iloc[i].values[0:4]
        temp = []
        for j in range(num_data_frame_row):
            y = data.iloc[j].values[0:4]
            temp.append(((np.sqrt(np.sum((x - y) ** 2)),j) ) )
        print temp
        eculidean_dis.append(temp)
    prepare_output(eculidean_dis, k, '../DataSet/Euclidean.csv')


def prepare_output(distance_matrix , k,filename):
    num_data_frame_row = len(distance_matrix)

    colnames = []
    for i in range(k - 1):
        colnames.append(str(i + 1))
        colnames.append(str(i + 1) + '-Prox')
    df = DataFrame(columns=colnames)

    # print the result
    for i in range(num_data_frame_row):
        # sort the tuple based on the euclidean_distance
        result = np.array(sorted(distance_matrix[i], key=lambda x: x[0]))
        l = []
        for j in range(k - 1):
            l.append(result[j + 1][1])
            l.append(result[j + 1][0])
        df.loc[i] = l
    df.columns.name = "Transaction ID"
    df.index += 1
    df.to_csv(path_or_buf=filename)


# Read the data file.
# fileName - The name of the file.
def read_data_file(fileName):
    try:
        # read the file
        data = pd.read_csv(fileName)
        data.shape
    except IOError:
        print ("Could not open file: "), fileName
    return data

def normalize_data(data):
    data = data[['sepal_length','sepal_width', ' petal_length',' petal_width']]
    print "data max", data.min()
    norm_data = (data - data.min()) / (data.max() - data.min())
    return norm_data

# the main function
def analyze_data():
    # fileName = input("Enter he data set path: ")
    fileName = '../DataSet/Iris.csv'
    #fileName = '..\DataSet\Iris.csv'
    data = read_data_file(fileName)
    #print type(data)
    #print data.head()
#    print "1. Cosine Similarity"
#    print "2. "
#    choice = input("Enter your choice:")

#    if choice == 1:
        # call the particular function
#    ellif choice == 2:
        #
#    else:
#        print "Error in choosing"
    #print data.head()
    data = normalize_data(data)
    euclidean_distance(data, 5)
    minkowski_distance(data, 5, 6)

# call the runner function.
analyze_data()
