# Project 1
#
# Author :  Rajarshi Biswas
#           Sayam Ganguly
import pandas as pd
import numpy as np

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
    #fileName = '../DataSet/Iri.csv'
    fileName = '..\DataSet\Iris.csv'
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
    cosine_similarity(data)
    return

# call the runner function.
analyze_data()
