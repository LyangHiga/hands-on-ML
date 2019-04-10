#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import os

import matplotlib as mpl
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_openml

from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import cross_val_score


from sklearn.preprocessing import StandardScaler




np.random.seed(42)

mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)

#The dataset isnt sorted like in the book, to organize like that:
def sort_by_target(mnist):
    reorder_train = np.array(sorted([(target, i) for i, target in enumerate(mnist.target[:60000])]))[:, 1]
    reorder_test = np.array(sorted([(target, i) for i, target in enumerate(mnist.target[60000:])]))[:, 1]
    mnist.data[:60000] = mnist.data[reorder_train]
    mnist.target[:60000] = mnist.target[reorder_train]
    mnist.data[60000:] = mnist.data[reorder_test + 60000]
    mnist.target[60000:] = mnist.target[reorder_test + 60000]


mnist = fetch_openml('mnist_784', version=1, cache=True)
# fetch_openml() returns targets as strings
mnist.target = mnist.target.astype(np.int8) 
# fetch_openml() returns an unsorted dataset
sort_by_target(mnist) 

#print(mnist["data"], mnist["target"])
#print(mnist.data.shape)


X, y = mnist["data"], mnist["target"]
# 70k images, 28 x 28 [0,255] pixel's intensity: 784
#print(X.shape)
#a label for each image
#print(y.shape)
#this digit is a five
some_digit = X[36000]
#plot_image(X)

#This dataset is already slpitted in training and test set
X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]
sgd_clf = SGDClassifier(max_iter=5, tol=-np.infty, random_state=42)

#sgd clf uses the One-Versus-All strategy
#train 10 binary classifiers( one for each class) then it take the class with the higher score
sgd_clf.fit(X_train, y_train)
print("X[36000] is a ",sgd_clf.predict([some_digit]))

#here we can see the score to each class
some_digit_scores = sgd_clf.decision_function([some_digit])
print('Scores = ',some_digit_scores)

#we can see what index represents each class
print('Index to class: ', sgd_clf.classes_)

#we can also use the RandomForestClassifier, it doesnt need to use any special strategy because 
#it can directly classify multiple classes
forest_clf = RandomForestClassifier(n_estimators=10, random_state=42)
forest_clf.fit(X_train, y_train)
print("X[36000] is a ", forest_clf.predict([some_digit]))
#we can see the probs of an instance in
print("X[36000] probs: ", forest_clf.predict_proba([some_digit]))

#lets evaluate with cross validation
print('sgd_clf Accuracy using CV with 3 folds: ', cross_val_score(sgd_clf, X_train, y_train, cv=3, scoring="accuracy"))
print('forest_clf Accuracy using CV with 3 folds: ', cross_val_score(forest_clf, X_train, y_train, cv=3, scoring="accuracy"))

#just making scaling we can get better 
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train.astype(np.float64))

cross_val_score(sgd_clf, X_train_scaled, y_train, cv=3, scoring="accuracy")

print('sgd_clf Accuracy using CV with 3 folds and StandardScaler: ', cross_val_score(sgd_clf, X_train_scaled, y_train, cv=3, scoring="accuracy"))
print('forest_clf Accuracy using CV with 3 folds and StandardScaler: ', cross_val_score(forest_clf, X_train_scaled, y_train, cv=3, scoring="accuracy"))