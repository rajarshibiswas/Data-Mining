import numpy as np
import pandas as pd
from pandas import DataFrame
#from sklearn.neighbors import KNeighborsClassifier
import warnings


def analyze_income_data(k):
    warnings.filterwarnings("ignore")
    filename = 'income_tr.csv'
    data = read_data_file(filename)
    test_data_filename = 'income_te.csv'
    test_data = read_data_file(test_data_filename)
    data = prepare_data(data)
    test_data = prepare_data(test_data)
    KNN_from_scratch(data, test_data, n_neighbors=40)
    #euclidean_distance(data, k+1)
    #cosine_similarity(data, k+1)


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
    #data.drop('class', axis=1, inplace=True)


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


# data = Data that used to classify unseen records.
# test_data = Data that needs to be classified.
# n_neighbors = number of neighbors to consider.
def KNN_from_scratch(data, test_data, n_neighbors):
    # Get the number of rows in the test data set
    test_data_row = test_data.shape[0]
    right_prediction = 0
    miss_prediction = 0
    colNames = ['Transaction ID','Actual Class', 'Predicted Class', 'Posterior Probability']
    output_frame = pd.DataFrame(columns=colNames)
    for i in range(test_data_row):
        neighbors = euclidean_distance(data, test_data.iloc[i], n_neighbors)
        result_class = []
        for j in range(len(neighbors)):
            result_class.append(data.iloc[neighbors[j][1]]['class'])
        predicted_class = get_predicted_class(result_class)
        output_frame.loc[i] = [i+1, test_data.iloc[i]["class"], predicted_class, 0.5]
        #output_frame.append(pd.Series([i, test_data.iloc[i]["class"], predicted_class, 0.5]),ignore_index=True)
        #print "Row %d, Predicted class %s, Actual class %s " %(i, predicted_class, test_data.iloc[i]["class"])
        if (predicted_class == test_data.iloc[i]["class"]):
            right_prediction+= 1
        else:
            miss_prediction+= 1
    output_frame.to_csv(path_or_buf='K-NN_Euclidean.csv')
    error_rate = ((miss_prediction * 100) / (right_prediction + miss_prediction))

    print "The error rate %f percent" %error_rate

        #print result_class


# Result class - The class of all K nearest neighbors
def get_predicted_class(result_class):
    c1 = [" <=50K", 0]
    c2 = [" >50K", 0]

    #print result_class
    for i in range(len(result_class)):
        if (result_class[i].find(" <=50K") != -1):
            c1[1] = c1[1] + 1
        elif (result_class[i].find(" >50K") != -1):
            c2[1] = c2[1] + 1
    c = [c1, c2]

    expected_class = sorted(c, key = lambda x: x[1])

#    print "The expected class:"
    return expected_class[1][0]
    #return expected_class


# Cosine Similarity proximity function.
def cosine_similarity(data, test_row, n_neighbors):
    dataRows = data.shape[0]
    #print data.iloc[1:3, 0:11]
    x = test_row.values[0:11]
    dotMatrix = data.iloc[:, 0:11].dot(x.T)
    #print np.sum(np.array(data.iloc[0, 0:11]**2))
    cos = []
    for j in range(dataRows):
        if j in data.index:
            # cos[i][j] = dotMatrix[i][j]/(np.sqrt(dotMatrix[i][i]) * np.sqrt(dotMatrix[j][j]))
            val = dotMatrix[j] / (np.sqrt(np.sum(np.array(x)**2)) * np.sqrt(np.sum(np.array(data.iloc[j, 0:11]**2))))
            cos.append((val, j))
        else:
            cos.append((0.0, j))
    result = (sorted(cos, key=lambda x: x[0], reverse=True))
    return result[0:n_neighbors]
    #prepare_output(cos, k, 'Income_Cosine.csv', True)


# Compute euclidean distance
def euclidean_distance(data, test_row, n_neighbors):
    # Get the number of rows in data
    num_data_frame_row = data.shape[0]

    # The result array
    euclidean_dis = []
    x = test_row.values[0:11]
    for j in range(num_data_frame_row):
        y = data.iloc[j].values[0:11]
        euclidean_dis.append((np.sqrt(np.sum((x - y) ** 2)), j))
    result = (sorted(euclidean_dis, key=lambda x: x[0]))
    return result[0:n_neighbors]
    #prepare_output(euclidean_dis, k, 'Income_Euclidean.csv', False)


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


analyze_income_data(6)
