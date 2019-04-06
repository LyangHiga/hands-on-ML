#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import os

import matplotlib as mpl
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_openml

from sklearn.linear_model import SGDClassifier

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

sgd_clf.fit(X_train, y_train)
print("X[36000] is a ",sgd_clf.predict([some_digit]))