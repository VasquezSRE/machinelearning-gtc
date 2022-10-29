import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from imblearn.metrics import geometric_mean_score

class Utils2 ():
    def __init__(self):
            pass
            
    def get_matrixes(self, route):
        return get_matrixes(route)
    
    def get_train_test(self, route):
        return get_train_text(route)
    
    def get_training_test(self, X, Y, train, test):
        return get_training_test(X, Y, train, test)
    
    def get_metrics(self, model, X_train, X_test, y_train, Ytest):
        return get_metrics(model, X_train, X_test, y_train, Ytest)
    
    def get_kernel_metrics(self, model, X_train, X_test, y_train, Ytest):
        return get_kernel_metrics(model, X_train, X_test, y_train, Ytest)
    
    def get_means_and_ic(self, f1, gmean, eficiencia_train, eficiencia_test):
        return get_means_and_ic(f1=f1, gmean=gmean, eficiencia_train=eficiencia_train, eficiencia_test=eficiencia_test)

def read_csv(route):
    return pd.read_csv(route, header=None)

def get_matrixes(route):
    df = read_csv(route)
    data = df.to_numpy()
    X = data[:,0:20]
    Y = data[:,20]
    return X, Y

def get_train_text(route):
    X, Y = get_matrixes(route)
    X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state=0, stratify=Y, shuffle=True, test_size = 0.25)
    return X_train, X_test, y_train, y_test

def get_training_test(X, Y, train, test):
    Xtrain = X[train,:]
    Ytrain = Y[train]
    Xtest = X[test,:]
    Ytest = Y[test]
    return Xtrain, Ytrain, Xtest, Ytest

def get_metrics(model, X_train, X_test, y_train, Ytest):
    Yest = model.predict(X_test)
    Ytrain_pred = model.predict(X_train)
    f1_score_current = f1_score(y_true = Ytest, y_pred=Yest, average = "weighted")
    gmean_current = geometric_mean_score(y_true = Ytest, y_pred=Yest, average="weighted")
    eficiencia_train_current = np.mean(Ytrain_pred.ravel() == y_train.ravel())
    eficiencia_test_current = np.mean(Yest.ravel() == Ytest.ravel())
    return f1_score_current, gmean_current, eficiencia_train_current, eficiencia_test_current

def get_kernel_metrics(model, X_train, X_test, y_train, Ytest):
    Yest = model.score_samples(X_test)
    Ytrain_pred = model.score_samples(X_train)
    f1_score_current = f1_score(y_true = Ytest, y_pred=Yest, average = "weighted")
    gmean_current = geometric_mean_score(y_true = Ytest, y_pred=Yest, average="weighted")
    eficiencia_train_current = np.mean(Ytrain_pred.ravel() == y_train.ravel())
    eficiencia_test_current = np.mean(Yest.ravel() == Ytest.ravel())
    return f1_score_current, gmean_current, eficiencia_train_current, eficiencia_test_current

def get_means_and_ic(f1, gmean, eficiencia_train, eficiencia_test):
    f1_mean = np.mean(f1)
    f1_ic = np.std(f1)
    gmean_mean = np.mean(gmean)
    gmean_ic = np.std(gmean)
    eficiencia_train_mean = np.mean(eficiencia_train)
    eficiencia_train_ic = np.std(eficiencia_train)
    eficiencia_test_mean = np.mean(eficiencia_test)
    eficiencia_test_ic = np.std(eficiencia_test)
    return f1_mean, f1_ic, gmean_mean, gmean_ic, eficiencia_train_mean, eficiencia_train_ic, eficiencia_test_mean, eficiencia_test_ic
    