import os as os
import pandas as pd
import numpy as np
import copy as copy
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull


def read_data():
    # Take the dataset as input
    filename = "wine.csv"
    try:
        # read the file
        data = pd.read_csv(filename)
        X, Y, cls = separate_data(data)
        k = data['quality'].unique().shape[0]
        #k = 4
        X = normalize_data(X)
        centroids = np.random.random((k, X.shape[1]))
        clusters, centroids = run_k_means(X, centroids, k)
        sse, true_ssb, pred_ssb = calculate_sse_ssb(X, Y, clusters, 
                                                    centroids, k)
        print ("True SSB =", true_ssb)
        print ("Predicted SSB =", pred_ssb)
        print (sse)
        print ("Predicted ")
        print ("True Total = ", true_ssb + sse['True SSE'].sum())
        print ("Pred Total = ", pred_ssb + sse['Predicted SSE'].sum())
        print (clusters.value_counts())
        d = plot(X, Y, clusters, k)
        print (pd.crosstab(d["True"],d["Predicted"]))
        output_frame = pd.DataFrame(columns=["ID","Cluster"])
        output_frame["ID"] = d["ID"]
        output_frame["Cluster"] = d["Predicted"]
        filename = filename[:-4] + "_output" + ".csv"
        output_frame.to_csv(path_or_buf= filename)

    except IOError:
        print("Could not open file: ", filename)


def separate_data(data):
    X = data.ix[:, 1:-2]
    Y = data['quality']
    cls = data['class']
    return X, Y, cls


def normalize_data(data):
    norm_data = data
    norm_data = (norm_data - norm_data.min()) / (norm_data.max() - norm_data.min())
    return norm_data


def run_k_means(X, cen, k):
    clusters = pd.DataFrame(np.zeros((X.shape[0], 2)), columns=[['Prev', 'Curr']])

    while is_changed(clusters):
        print ("Running")
        clusters["Prev"] = clusters["Curr"]
        cen = move_centroids(X, clusters["Prev"], cen, k)
        for i in range(X.shape[0]):
            dist = []
            for j in range(cen.shape[0]):
                x = X.values[i]
                y = cen[j]
                dist.append(np.sqrt(np.sum((x - y) ** 2)))
            clusters.loc[i, "Curr"] = dist.index(min(dist)) + 1
    return clusters["Curr"], cen


def is_changed(clusters):
    if clusters.loc[0, "Curr"] == 0:
        return True
    changed = clusters.loc[clusters["Curr"] != clusters["Prev"]]
    if changed.shape[0] == 0:
        return False
    else:
        return True


def move_centroids(X, clusters, cen, k):
    if clusters[0] == 0:
        return cen
    for i in range(k):
        ind = list(np.where(clusters == i+1)[0])
        if len(ind) != 0:
            cen[i] = X[X.index.isin(ind)].sum() / len(ind)
        else:
            cen[i] = [0] * X.shape[1]
    return cen


def calculate_sse_ssb(X, Y, clusters, centroids, k):
    sse = pd.DataFrame(np.zeros((k, 3)), columns=[['Cluster', 'True SSE', 'Predicted SSE']])
    true_cen = move_centroids(X, Y, np.zeros((k, X.shape[1])), k)
    true_cen_mean = true_cen.mean()
    pred_cen_mean = centroids.mean()
    true_ssb = 0
    pred_ssb = 0
    for i in range(k):
        sse.loc[i, "Cluster"] = i+1
        ind = list(np.where(clusters == i + 1)[0])
        sse.loc[i, "Predicted SSE"] = get_sse(centroids[i], X[X.index.isin(ind)])
        pred_ssb += np.sum(len(ind)*(pred_cen_mean - centroids[i]) ** 2)
        ind = list(np.where(Y == i + 1)[0])
        sse.loc[i, "True SSE"] = get_sse(true_cen[i], X[X.index.isin(ind)])
        true_ssb += np.sum(len(ind)*(true_cen_mean - true_cen[i]) ** 2)
    return sse, true_ssb, pred_ssb


def get_sse(cen, members):
    sum = 0
    if members.shape[0] == 0:
        return sum
    for i in range(members.shape[0]):
        x = members.values[i]
        sum += np.sum((x - cen) ** 2)
    return sum


def plot(X, Y, clusters, k):
    d = pd.DataFrame(columns=[["ID", "X.1", "X.2", "True", "Predicted"]])
    d["X.1"] = X["alcohol"]
    d["X.2"] = X["sulph"]
    d["True"] = Y
    d["Predicted"] = clusters
    d["ID"] = d.index + 1
    plt.figure()
    jet = plt.get_cmap('prism')
    colors = iter(jet(np.linspace(0,1,10)))
    for i in range(k):
        df_true = d.loc[d["True"] == i+1]
        df_pred = d.loc[d["Predicted"] == i+1]
        #hull_true = ConvexHull(df_true[["X.1", "X.2"]].as_matrix())
        #hull_pred = ConvexHull(df_pred[["X.1", "X.2"]].as_matrix())
        color = next(colors)
        plt.subplot(1,2,1)
        plt.scatter(df_true["X.1"], df_true["X.2"], marker=".", color = color)
        plt.subplot(1,2,2)
        plt.scatter(df_pred["X.1"], df_pred["X.2"], marker=".", color = color)
        #for simplex in hull_true.simplices:
            #plt.plot(df_true.ix[simplex, "X.1"], df_true.ix[simplex, "X.2"], 'b-')
        #for simplex in hull_pred.simplices:
         #   plt.plot(df_pred.ix[simplex, "X.1"], df_pred.ix[simplex, "X.2"], 'r-')
    plt.show()
    return d



read_data()
