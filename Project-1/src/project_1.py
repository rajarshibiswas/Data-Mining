# Project 1
#
# Author :  Rajarshi Biswas
#           Sayam Ganguly
import pandas as pd
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
    # The result array
    eculidean_dis = np.zeros((num_data_frame_row, num_data_frame_row))

    for i in range(num_data_frame_row):
        x = data.iloc[i].values[0:4]
        for j in range(num_data_frame_row):
            y = data.iloc[j].values[0:4]
            eculidean_dis[i][j] = np.sqrt(np.sum((x - y) ** 2))

    # print the result
    for i in range(num_data_frame_row):
        y = np.sort(eculidean_dis[i])
        print y[1:k+1]
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
