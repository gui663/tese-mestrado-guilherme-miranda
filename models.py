from random import random
import pandas as pd
import numpy as np
from sklearn.metrics import PrecisionRecallDisplay
import matplotlib.pyplot as plt

#SVM
from numpy import mean
from numpy import std
from sklearn import svm
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_curve
import sklearn.metrics as metrics
from sklearn.model_selection import KFold


def kf(model, random_state, model_name, X, y):

    
    print(model_name + " k-Fold CV")

    for n in (2,5,10):

        cv = KFold(n_splits=n, random_state=random_state, shuffle=True)
        scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
        print('Accuracy with %i folds: %.2f (%.2f)' % (n, mean(scores), std(scores)))

    

def SVM(X,y):

    print("===== SVM Training =====")
    random_state = 21
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25,random_state=random_state)

    #fs = SelectKBest(score_func=f_classif, k=5)

    #data_selected = fs.fit_transform(X, y)
    #data_selected = pd.DataFrame(data_selected)
    #init: gamma=25 C=100
    clf = svm.SVC(kernel='rbf', gamma=100, C =25)
    clf.fit(X_train, y_train)
    predictions  = clf.predict(X_test)
    print("Accuracy score of RBF SVM with no K-fold: ", round(accuracy_score(y_test, predictions), 2), '%')
    print(classification_report(y_test,predictions))
    kf(clf, random_state, "SVM", X, y)
    print("==========")
    
    
    fig, ax = plt.subplots()

    display = PrecisionRecallDisplay.from_estimator(
    clf, X_test, y_test, name="RBFSVC", ax=ax, color='teal')
    ax.set_ylabel('Precision')
    ax.set_xlabel('Recall')
    plt.show()

    fpr, tpr, threshold = metrics.roc_curve(y_test, predictions)
    roc_auc = metrics.auc(fpr, tpr)
    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, color='teal', label = 'AUC = %0.2f' % roc_auc)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1], color='forestgreen', linestyle='--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('Taxa de Verdadeiros Positivos')
    plt.xlabel('Taxa de Falsos Positivos')
    plt.show()
    #precision, recall, _ = precision_recall_curve(y_test, predictions)
    #disp = PrecisionRecallDisplay(precision=precision, recall=recall)
    #scores = cross_val_score(clf, X, y, cv=10)
    #print(scores)
    #print(scores.mean())

#NN
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import tensorflow as tf
def NN(X, y):
    print("===== NN Training =====")
    random_state=42
    shape = len(X.columns)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=random_state)

    model = Sequential()
    model.add(Dense(36, input_shape=(shape,), activation='relu'))
    model.add(Dense(24, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy', tf.keras.metrics.AUC(), tf.keras.metrics.Recall(), tf.keras.metrics.Precision()])

    history = model.fit(X_train, y_train, epochs=150, batch_size=10)
    
    _, accuracy, auc, precision, recall = model.evaluate(X_test, y_test)
    print('Accuracy score of NN: %.2f' % (accuracy*100))
    print("==========")

    disp = PrecisionRecallDisplay(precision=history.history['precision'], recall=history.history['recall'])
    disp.plot()
    plt.title('AUC ' + str(auc))
    plt.show()

# LOGISTIC REGRESSION
# https://www.analyticssteps.com/blogs/introduction-logistic-regression-sigmoid-function-code-explanation
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

def logistic_regression(X, y):
    
    x_train_lr, x_test_lr, y_train_lr, y_test_lr = train_test_split(X, y, test_size = 0.25, random_state = 0)

    classifier_lr = LogisticRegression(random_state=0)
    classifier_lr.fit(x_train_lr, y_train_lr)

    y_pred_lr = classifier_lr.predict(x_test_lr)
    accuracy_score_lr = accuracy_score(y_test_lr, y_pred_lr)
    print("Accuracy score of logistic regression: ", round(accuracy_score_lr,2), '%')

# Random Forests

from sklearn.ensemble import RandomForestClassifier

def RF(X, y):

    print("===== RF Training =====")
    random_state = 21
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25,random_state=random_state)

    clf = RandomForestClassifier(random_state=21)

    clf.fit(X_train, y_train)
    predictions  = clf.predict(X_test)
    print("Accuracy score of random forests with no K-fold: ", round(accuracy_score(y_test, predictions), 2), '%')
    kf(clf, random_state, "RF", X, y)
    print("==========")

    fig, ax = plt.subplots()

    display = PrecisionRecallDisplay.from_estimator(
    clf, X_test, y_test, name="RF", ax=ax, color='teal')

    ax.set_ylabel('Precision')
    ax.set_xlabel('Recall')
    plt.show()

    fpr, tpr, threshold = metrics.roc_curve(y_test, predictions)
    roc_auc = metrics.auc(fpr, tpr)
    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, color='teal', label = 'AUC = %0.2f' % roc_auc)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1], color='forestgreen', linestyle='--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('Taxa de Verdadeiros Positivos')
    plt.xlabel('Taxa de Falsos Positivos')
    plt.show()

#kNN
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

def kNN(X, y):

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25,random_state=21)

    clf = KNeighborsRegressor(n_neighbors=5)

    clf.fit(X_train, y_train)
    predictions  = clf.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    rmse = mean_squared_error(y_test, predictions, squared=False)
    print("Accuracy score of random forests: ", "mae: ", mae, " mse: ", mse, " rmse: ", rmse)