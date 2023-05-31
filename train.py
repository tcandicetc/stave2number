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
    OneVsOneModel = OneVsOneClassifier(SVC())
    OneVsOneModel.fit(X_train, y_train)
    start_time = time()
    y_pred = OneVsOneModel.predict(X_test)
    end_time = time()
    print("-----One Vs One Model-----")
    print(f"Error rate: {((y_test != y_pred).sum() / len(X_test) * 100): .2f}%")
    print(f"Testing time: {(end_time - start_time): .2f} s")
    
    with gzip.GzipFile('model/OneVsOneModel.pgz', 'w') as f:
        pickle.dump(OneVsOneModel, f)

def trainingOneVsRestModel(X_train, X_test, y_train, y_test):
    OneVsRestModel = OneVsRestClassifier(SVC())
    OneVsRestModel.fit(X_train, y_train)
    start_time = time()
    y_pred = OneVsRestModel.predict(X_test)
    end_time = time()
    print("-----One Vs Rest Model-----")
    print(f"Error rate: {((y_test != y_pred).sum() / len(X_test) * 100): .2f}%")
    print(f"Testing time: {(end_time - start_time): .2f} s")

    with gzip.GzipFile('model/OneVsRestModel.pgz', 'w') as f:
        pickle.dump(OneVsRestModel, f)

def trainingKNeighborsModel(X_train, X_test, y_train, y_test):
    KNeighborsModel = KNeighborsClassifier(n_neighbors=3)
    KNeighborsModel.fit(X_train, y_train)
    start_time = time()
    y_pred = KNeighborsModel.predict(X_test)
    end_time = time()
    print("-----K Neighbors Model-----")
    print(f"Error rate: {((y_test != y_pred).sum() / len(X_test) * 100): .2f}%")
    print(f"Testing time: {(end_time - start_time): .2f} s")

    with gzip.GzipFile('model/KNeighborsModel.pgz', 'w') as f:
        pickle.dump(KNeighborsModel, f)

if __name__ == '__main__':
    X_1 = np.load('dataset/datasetX_1.npy')
    X_2 = np.load('dataset/datasetX_2.npy')
    X_3 = np.load('dataset/datasetX_3.npy')
    X = np.concatenate((X_1, X_2, X_3))
    newX = []
    for i in range(len(X)):
        newX.append(np.ravel(X[i]))
    X = newX
    y_1 = np.load('dataset/datasetY_1.npy')
    y_2 = np.load('dataset/datasetY_2.npy')
    y_3 = np.load('dataset/datasetY_3.npy')
    y = np.concatenate((y_1, y_2, y_3))
    for i in range(16):
        print(f"sum of label {i} image: {np.count_nonzero(y == i)}")
    print()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, train_size=0.8, random_state=0, stratify=y)

    trainingOneVsOneModel(X_train, X_test, y_train, y_test)
    trainingOneVsRestModel(X_train, X_test, y_train, y_test)
    trainingKNeighborsModel(X_train, X_test, y_train, y_test)