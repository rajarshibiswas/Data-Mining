# Project 1
#
# Author :  Rajarshi Biswas
#           Sayam Ganguly
import pandas as pd
from pandas import DataFrame
import numpy as np
np.set_printoptions(precision=3)

# Cosine Similarity proximity function.
def cosine_similarity(data):
    # Algo for Cosine Similarity
    #dataArr = data.as_matrix()[:,:-1]
    #print dataArr
    df = data.loc[:,['sepal_length','sepal_width',' petal_length',
    ' petal_width']].copy()
    dataRows = df.shape[0]
    print df.shape[1]
    dotMatrix = df.dot(df.T)
    #print dotMatrix.head()
    cos = np.zeros((dataRows,dataRows-1))
    for i in range(dataRows-1):
        for j in range(dataRows-2):
            if i!=j:
                cos[i][j] = dotMatrix[i][j]/(np.sqrt(dotMatrix[i][i]) * np.sqrt(dotMatrix[j][j]))
    print cos[1]
    #print data.head()
    return

# Compute eculidean distance
def euclidean_distance(data, k):
    # Get the number of rows in data
    num_data_frame_row = data.shape[0]
    num_data_frame_row = 5
    # The result array
    eculidean_dis = []
    for i in range(num_data_frame_row):
        x = data.iloc[i].values[0:4]
        temp = []
        for j in range(num_data_frame_row):
            y = data.iloc[j].values[0:4]
            temp.append(((np.sqrt(np.sum((x - y) ** 2)),j) ) )
        eculidean_dis.append(temp)

    df = DataFrame(columns=('1st', '1s-pre', '2nd', '2nd-pre', '3rd', '3rd-pre'))

    # print the result
    for i in range(num_data_frame_row):
        # sort the tuple based on the euclidean_distance
        result =  np.array(sorted(eculidean_dis[i], key = lambda x:x[0]))
        #df.loc[i] = [result[0][1:k+1]]
        l = []
        for k in range(4):
            l.append(result[k+1][0])
            l.append(result[k+1][1])
        df.loc[i] = [l[1],l[0],l[3],l[2],l[5],l[6]]
        #print result[1][1]
        #print result[1:k+1]
    #print result
    #resultDF = DataFrame(data = result)
    df.index.name = "ID"
    print df
    return


# Read the data file.
# fileName - The name of the file.
def read_data_file(fileName):
    try:
        # read the file
        data = pd.read_csv(fileName)
    except IOError:
        print ("Could not open file: "), fileName
    return data

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
    euclidean_distance(data, 4)
    return

# call the runner function.
analyze_data()
