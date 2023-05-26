from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsOneClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from time import time
import pickle
import gzip

def trainingOneVsOneModel(X_train, X_test, y_train, y_test):
    start_time = time()
    OneVsOneModel = OneVsOneClassifier(SVC())
    y_pred = OneVsOneModel.fit(X_train, y_train).predict(X_test)
    end_time = time()
    print("-----One Vs One Model-----")
    print("Number of mislabeled points out of a total %d points : %d" % (len(X_test), (y_test != y_pred).sum()))
    print("Running time = %.2f s" % (end_time - start_time))
    
    with gzip.GzipFile('OneVsOneModel.pgz', 'w') as f:
        pickle.dump(OneVsOneModel, f)

def trainingOneVsRestModel(X_train, X_test, y_train, y_test):
    start_time = time()
    OneVsRestModel = OneVsRestClassifier(SVC())
    y_pred = OneVsRestModel.fit(X_train, y_train).predict(X_test)
    end_time = time()
    print("-----One Vs Rest Model-----")
    print("Number of mislabeled points out of a total %d points : %d" % (len(X_test), (y_test != y_pred).sum()))
    print("Running time = %.2f s" % (end_time - start_time))

    with gzip.GzipFile('OneVsRestModel.pgz', 'w') as f:
        pickle.dump(OneVsRestModel, f)

def trainingKNeighborsModel(X_train, X_test, y_train, y_test):
    start_time = time()
    KNeighborsModel = KNeighborsClassifier(n_neighbors=3)
    y_pred = KNeighborsModel.fit(X_train, y_train).predict(X_test)
    end_time = time()
    print("-----K Neighbors Model-----")
    print("Number of mislabeled points out of a total %d points : %d" % (len(X_test), (y_test != y_pred).sum()))
    print("Running time = %.2f s" % (end_time - start_time))

    with gzip.GzipFile('KNeighborsModel.pgz', 'w') as f:
        pickle.dump(KNeighborsModel, f)

if __name__ == '__main__':
    X_1 = np.load('datasetX_1.npy')
    X_2 = np.load('datasetX_2.npy')
    X_3 = np.load('datasetX_3.npy')
    X = np.concatenate((X_1, X_2, X_3))
    newX = []
    for i in range(len(X)):
        newX.append(np.ravel(X[i]))
    X = newX
    y_1 = np.load('datasetY_1.npy')
    y_2 = np.load('datasetY_2.npy')
    y_3 = np.load('datasetY_3.npy')
    y = np.concatenate((y_1, y_2, y_3))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, train_size=0.8, random_state=0, stratify=y)

    trainingOneVsOneModel(X_train, X_test, y_train, y_test)
    trainingOneVsRestModel(X_train, X_test, y_train, y_test)
    trainingKNeighborsModel(X_train, X_test, y_train, y_test)