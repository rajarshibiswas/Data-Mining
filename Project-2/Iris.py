# Project 2
# Iris data set
#
# Author :  Rajarshi Biswas
#           Sayam Ganguly

import pandas as pd
from pandas import DataFrame
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
np.set_printoptions(precision=3)

# Read the data file.
# fileName - The name of the file.
def read_data_file(fileName):
    try:
        # read the file
        data = pd.read_csv(fileName)
    except IOError:
        print ("Could not open file: "), fileName
    return data

# Normalize and returns the Iris dataset
# data - The Iris dataset
def normalize_train_data(data):
    norm_data = data
    norm_data = norm_data[['sepal_length','sepal_width', ' petal_length',' petal_width']]
    norm_data = (norm_data - norm_data.min()) / (norm_data.max() - norm_data.min())
    norm_data[' class'] = data[' class']
    return norm_data

def normalize_test_data(data):
    norm_data = data
    norm_data = norm_data[['sepal_length','sepal_width', 'petal_length','petal_width']]
    norm_data = (norm_data - norm_data.min()) / (norm_data.max() - norm_data.min())
    norm_data['class'] = data['class']
    return norm_data

def minkowski_distance(data, test_row, n_neighbors, r):
    num_data_frame_row = data.shape[0]
    # The result array
    minkowski_dis = []

    x = test_row.values[0:4]

    for j in range(num_data_frame_row):
        y  = data.iloc[j].values[0:4]
        minkowski_dis.append((np.power(np.sum(np.power(np.absolute(x - y), r)), (1.0/r)), j))

    result = (sorted(minkowski_dis, key = lambda x: x[0]))

    return (result[0:n_neighbors])

# Compute euclidean distance for Iris dataset
# data  - the data set
# k     -
def euclidean_distance(data, test_row, n_neighbors):
    # Get the number of rows in data
    num_data_frame_row = data.shape[0]
    #num_data_frame_row  = 5

    # The result array
    euclidean_dis = []

    x = test_row.values[0:4]

    for j in range(num_data_frame_row):
        y = data.iloc[j].values[0:4]
        euclidean_dis.append(( (np.sqrt (np.sum ((x - y) ** 2)), j) ) )

        #    prepare_output(eculidean_dis, k, 'Iris_Euclidean.csv')

    result = (sorted(euclidean_dis, key = lambda x: x[0]))

    return result[0:n_neighbors]


    #print eculidean_dis

def get_predicted_class_weigh(result_class, distance):
    c1 = ["Iris-setosa", 0]
    c2 = ["Iris-versicolor", 0]
    c3 = ["Iris-virginica", 0]

    #print result_class
    for i in range(len(result_class)):
        print distance[i]
        if (result_class[i].find("setosa") != -1):
            c1[1] = c1[1] + (1.0 / (distance[i] * distance[i]) )
        elif (result_class[i].find("versicolor") != -1):
            c2[1] = c2[1] + (1.0 / (distance[i] * distance[i]) )
        else:
            c3[1] = c3[1] + (1.0 / (distance[i] * distance[i]) )
    c = [c1, c2, c3]

    expected_class = sorted(c, key = lambda x: x[1])
    pos_prob = expected_class[2][1]

    # send back the expected class and calculated probability.
    result = [expected_class[2][0], pos_prob]
    return result


# Result class - The class of all K nearest neighbors
def get_predicted_class_majority_vote(result_class):
    c1 = ["Iris-setosa", 0]
    c2 = ["Iris-versicolor", 0]
    c3 = ["Iris-virginica", 0]

    #print result_class
    for i in range(len(result_class)):
        if (result_class[i].find("setosa") != -1):
            c1[1] = c1[1] + 1
        elif (result_class[i].find("versicolor") != -1):
            c2[1] = c2[1] + 1
        else:
            c3[1] = c3[1] + 1
    c = [c1, c2, c3]

    expected_class = sorted(c, key = lambda x: x[1])
    pos_prob = ( (float)(expected_class[2][1]) / len(result_class)  )

    # send back the expected class and calculated probability.
    result = [expected_class[2][0], pos_prob]
    return result


# data = Data that used to classify unseen records.
# test_data = Data that needs to be classified.
# n_neighbors = number of neighbors to consider.
def KNN_from_scratch(data, test_data, n_neighbors):
    # Get the number of rows in the test data set
    test_data_row = test_data.shape[0]
    #test_data_row =
    right_prediction = 0
    miss_prediction = 0
    colNames = ['Transaction ID','Actual Class', 'Predicted Class', 'Posterior Probability']
    output_frame = pd.DataFrame(columns=colNames)

    for i in range(test_data_row):
        #neighbors = euclidean_distance(data, test_data.iloc[i], n_neighbors)
        neighbors = minkowski_distance(data, test_data.iloc[i], n_neighbors, 5)
        result_class = []  # Store the classes of the nearest neighbors sorted by distacne.
        result_distance = [] # Store the correspoding distane.
        for j in range(len(neighbors)):
            result_class.append(data.iloc[neighbors[j][1]][' class'])
            result_distance.append(neighbors[j][0])
        #result = get_predicted_class_majority_vote(result_class)
        result = get_predicted_class_weigh(result_class, result_distance)

        #### Max #####
        predicted_class = result[0]
        output_frame.loc[i] = [i+1, test_data.iloc[i]["class"], predicted_class, result[1]]
        #print result[1]
        #print "Row %d, Predicted class %s, Actual class %s " %(i, predicted_class, test_data.iloc[i]["class"])
        if (predicted_class == test_data.iloc[i]["class"]):
            right_prediction+= 1
        else:
            miss_prediction+= 1
    output_frame.index = output_frame['Transaction ID']
    output_frame.drop('Transaction ID', axis = 1, inplace = True)

    output_frame['Posterior Probability'] = (output_frame['Posterior Probability'] - output_frame['Posterior Probability'].min()) / (output_frame['Posterior Probability'].max() - output_frame['Posterior Probability'].min())
    output_frame.to_csv(path_or_buf='IRIS_K-NN.csv')

    error_rate = ((miss_prediction * 100) / (right_prediction + miss_prediction))

    print "The error rate %f percent" %error_rate

        #print result_class


def KNN_using_scikit(data, test_data, n_neighbors):
    # Create the target data
    data.loc[data[' class'].str.contains("setosa"), 'c'] = 1
    data.loc[data[' class'].str.contains("versicolor"), 'c'] = 2
    data.loc[data[' class'].str.contains("virginica"), 'c'] = 3
    data[' class'] = data['c']
    data.drop('c', axis=1, inplace=True)
    target_data = data[[' class']]
    target_data = target_data[' class']
    target_data = target_data.as_matrix()

    # Create the training data
    train_data = data[['sepal_length','sepal_width', ' petal_length',' petal_width']]
    train = pd.DataFrame.as_matrix(train_data)

    neigh = KNeighborsClassifier(n_neighbors = n_neighbors)
    neigh.fit(train, target_data)
    right_prediction = 0
    miss_prediction = 0
    # number of rows in test dataset
    test_data_row = test_data.shape[0]
    for i in range(test_data_row):
        result = (neigh.predict([test_data.iloc[i].values[0:4]]))
        #print result
        if result == 1:
            #print "Predicted Class for row %d is Iris-setosa" %(i)
            if (test_data.iloc[i]["class"].find("setosa") != -1):
                test_data.iloc[i]["class"]
                right_prediction += 1
            else:
                miss_prediction += 1
        elif result == 2:
            #print "Expected Class for row %d is Iris-versicolor" %(i)
            if (test_data.iloc[i]["class"].find("versicolor") != -1):
                right_prediction += 1
            else:
                miss_prediction +=1
        elif result == 3:
            #print "Expected Class for row %d is Iris-virginica" %(i)
            if (test_data.iloc[i]["class"].find("virginica") != -1):
                right_prediction += 1
            else:
                miss_prediction += 1

    error_rate = ((miss_prediction * 100) / (right_prediction + miss_prediction))
    print "The error rate %f percent" %error_rate


    # create the target


#def KNN_from_scratch(data):

# The main function that starts the analysis
def analyze_iris_data(k):
    # Take the dataset as input
    data_fileName = 'Iris.csv'
    test_data_filename = 'Iris_Test.csv'
    data = read_data_file(data_fileName)
    test_data = read_data_file(test_data_filename)

    data = normalize_train_data(data)
    test_data = normalize_test_data(test_data)
    #test_data = normalize_data(test_data)
    # Normalize the data
    # calculate the distances.
    #minkowski_distance(data, k, 6)

    KNN_from_scratch(data, test_data, n_neighbors = 40)
    #KNN_using_scikit(data, test_data, n_neighbors=40)

analyze_iris_data(4)
