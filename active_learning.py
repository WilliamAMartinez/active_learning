#This script is to represent the active learning rutines

import sklearn
from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
import pandas as pd


def get_data():
    data = load_iris()
    X = data.data
    y = data.target
    print ('IRIS:', X.shape, y.shape)
    return (X, y)

#Normalization

class Normalize:

    print('Normalizing min-max')

    def normalize(self, X):
        self.scaler = MinMaxScaler()
        X_normal = self.scaler.fit_transform(X)
        return (X_normal)

#random selection

def split(size_train):

    X,y = get_data()
    X_n = Normalize()
    X_normal = X_n.normalize(X)
    #random index
    np.random.seed(1)
    index_random = np.random.choice(X_normal.shape[0],X_normal.shape[0], replace= False)
    #train
    X_train = X_normal[index_random[:size_train]]
    y_train = y[index_random[:size_train]]
    #test
    X_test = X_normal[index_random[size_train:]]
    y_test = y[index_random[size_train:]]
    return(X_train, y_train, X_test, y_test)


#Entropies

class measureselection():
    def EntropySelection(self, probabilities):
        def entropyfunc(x):
            #ind = np.where(x != 0)[0]
            e = -(x * np.log2(x)).sum()
            return(e)
        #implementing function
        entropy = np.apply_along_axis(entropyfunc, 1, probabilities)
        #fransforming entropie values in probabilities using sof-max function
        def softmax(y):
            delta = np.exp(y)/np.sum(np.exp(y))
            return delta
        self.entropy = softmax(entropy)
        print(self.entropy.shape)
        return(self.entropy)

class SvmModel():
    model_type = 'Support Vector Machine with linear Kernel'
    def fit_predict(self, X_val, y_val, X_train, X_test):
        print ('training svm...')
        self.classifier = SVC(C=2, kernel = 'rbf',gamma= 0.25, probability=True)
        self.classifier.fit(X_val, y_val)
        self.train_y_predicted = self.classifier.predict(X_train)
        self.test_y_predicted = self.classifier.predict(X_test)
        self.train_y_probability = self.classifier.predict_proba(X_train)
        return(self.train_y_predicted , self.train_y_probability, self.test_y_predicted)
