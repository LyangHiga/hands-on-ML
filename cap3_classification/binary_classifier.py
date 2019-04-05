# Common imports
import numpy as np
import os

# to make this notebook's output stable across runs
np.random.seed(42)

# To plot pretty figures
import matplotlib as mpl
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_openml
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import cross_val_score

from sklearn.base import BaseEstimator

from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import f1_score

from sklearn.metrics import precision_recall_curve

from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score

from sklearn.ensemble import RandomForestClassifier




mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)

# Where to save the figures
PROJECT_ROOT_DIR = "."
CHAPTER_ID = "classification"


#this is a dumb classifier, always say that is not a 5
class Never5Classifier(BaseEstimator):
    def fit(self, X, y=None):
        pass
    def predict(self, X):
        return np.zeros((len(X), 1), dtype=bool)

def save_fig(fig_id, tight_layout=True):
    path = os.path.join(PROJECT_ROOT_DIR, "images", CHAPTER_ID, fig_id + ".png")
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format='png', dpi=300)

#The dataset isnt sorted like in the book, to organize like that:
def sort_by_target(mnist):
    reorder_train = np.array(sorted([(target, i) for i, target in enumerate(mnist.target[:60000])]))[:, 1]
    reorder_test = np.array(sorted([(target, i) for i, target in enumerate(mnist.target[60000:])]))[:, 1]
    mnist.data[:60000] = mnist.data[reorder_train]
    mnist.target[:60000] = mnist.target[reorder_train]
    mnist.data[60000:] = mnist.data[reorder_test + 60000]
    mnist.target[60000:] = mnist.target[reorder_test + 60000]

def plot_image(X,some_digit):
	some_digit_image = some_digit.reshape(28, 28)
	plt.imshow(some_digit_image, cmap = mpl.cm.binary,
	           interpolation="nearest")
	plt.axis("off")

	plt.savefig("some_digit_plot")
	plt.show()

def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):
    plt.plot(thresholds, precisions[:-1], "b--", label="Precision", linewidth=2)
    plt.plot(thresholds, recalls[:-1], "g-", label="Recall", linewidth=2)
    plt.xlabel("Threshold", fontsize=16)
    plt.legend(loc="upper left", fontsize=16)
    plt.ylim([0, 1])
    plt.xlim([-700000, 700000])
    plt.figure(figsize=(8, 4))
    plt.savefig("precision_recall_vs_threshold")
    plt.show()   

def plot_precision_vs_recall(precisions, recalls):
    plt.plot(recalls, precisions, "b-", linewidth=2)
    plt.xlabel("Recall", fontsize=16)
    plt.ylabel("Precision", fontsize=16)
    plt.axis([0, 1, 0, 1])
    plt.figure(figsize=(8, 6))
    plt.savefig("precision_vs_recall_plot")
    plt.show()


def plot_roc_curve(fpr, tpr, label=None):
    plt.plot(fpr, tpr, linewidth=2, label=label)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.axis([0, 1, 0, 1])
    plt.xlabel('False Positive Rate', fontsize=16)
    plt.ylabel('True Positive Rate', fontsize=16)
    plt.figure(figsize=(8, 6))
    plt.savefig("roc_curve_plot")
    plt.show()

def roc_curve_comparison_plot(fpr, tpr, fpr_forest, tpr_forest):
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, "b:", linewidth=2, label="SGD")
    plot_roc_curve(fpr_forest, tpr_forest, "Random Forest")
    plt.legend(loc="lower right", fontsize=16)
    plt.savefig("roc_curve_comparison_plot")
    plt.show()



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
some_digit = X[36000]
#plot_image(X)

#This dataset is already slpitted in training and test set
X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]

#It's important so shuffle the data to do cross-validaton, we want that all folds contain all digits 
shuffle_index = np.random.permutation(60000)
X_train, y_train = X_train[shuffle_index], y_train[shuffle_index]

#We want to classify if a image is a 5 or not
y_train_5 = (y_train == 5)
y_test_5 = (y_test == 5)

sgd_clf = SGDClassifier(max_iter=5, tol=-np.infty, random_state=42)
sgd_clf.fit(X_train, y_train_5)

#print(sgd_clf.predict([some_digit]))

#Measuring the classifier by accuracy
#print(cross_val_score(sgd_clf, X_train, y_train_5, cv=3, scoring="accuracy"))
never_5_clf = Never5Classifier()
# We can see why accuracy sometimes is a bad mesuare. In this case 
#almost 90% of all images arent a 5. So this dumb classifier has a great score, even if he cant generalize.
#print(cross_val_score(never_5_clf, X_train, y_train_5, cv=3, scoring="accuracy"))

#Measuring the classifier by Confusion Matrix ( Precrision, Recall)
#to get the confusion matrix by training set using cross prediction
y_train_pred = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3)
#[0][0]: correctly pred od non 5 , [0][1]: non-5 wrong classified like a 5
#[1][0]: 5 wrong classified like a non-5, [1][1]: 5 correctly classified like a 5
print(confusion_matrix(y_train_5, y_train_pred))

#precision and recall
print("Precision = ", precision_score(y_train_5, y_train_pred))
print("Recall = ",recall_score(y_train_5, y_train_pred))

#Calculating Precision by harding code: count the number of True Positives and divide by all positives (classified like positive, true or false)
# this is like the accuracy from postive prediction
print("Precision = ", (4344.0 / (4344 + 1307)) )

#Doing the same for Recall: we have to count all the True positives and divide by all that actually are positives (classified or not like that)
print("Recall = ",(4344.0 / (4344 + 1077)) )

#A better way to measure a classifier is by the F1 Score, what means a harmonic mean between Precision and Recall
print("f1 = " , f1_score(y_train_5, y_train_pred))
# we can calculate the F1 score by harding code
print("f1 = " , 4344.0 / (4344.0 + (1077 + 1307)/2))

#lets check how scores and threshold work, we cant change the threshold directly so...
y_scores = sgd_clf.decision_function([some_digit])
print("score of X[36000] = ", y_scores)

#we can set our own threshold, for now lets use the same value that the scikit-learn
threshold = 0
y_some_digit_pred = (y_scores > threshold)

print("Is X[36000] a 5?" , y_some_digit_pred)

#lets rise the threshould, this way to some digit be pred like a 5 it needs a higher score (higher than the threshold)

threshold = 200000
y_some_digit_pred = (y_scores > threshold)
#so X[36000] hasnt a score great enough to be classified like a 5 anymore
print("Is X[36000] a 5? ", y_some_digit_pred)

#lets calculate all scores and take the precision recall curve
y_scores = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3, method="decision_function")
#with this func we get the tresholds values for each recall/precision 
precisions, recalls, thresholds = precision_recall_curve(y_train_5, y_scores)
#plot_precision_recall_vs_threshold(precisions, recalls, thresholds)

#now only the instances with a score bigger than 70k will be labeled as a 5
y_train_pred_90 = (y_scores > 70000)
print("Precision for y_train_pred_90 ",precision_score(y_train_5, y_train_pred_90))
print("Recall for y_train_pred_90 ",recall_score(y_train_5, y_train_pred_90))

#we can also plot precision x recall to see this trade off 
#plot_precision_vs_recall(precisions, recalls)

#ROC Curve and AUC

#We need the TPR (True Positive Rate - Recall) and the FPR(False positive Rate)
#FPR:  Ratio of negatives instacnces classified as positive

fpr, tpr, thresholds = roc_curve(y_train_5, y_scores)
#plot_roc_curve(fpr, tpr)

#AUC: Area under de curve, A perfect classifier has a AUC equal to 1, all graph is under the curve, therefore how much closer to 1 better.
print("AUC = ", roc_auc_score(y_train_5, y_scores))
#We can compare the AUC between differents classifiers
#Unfortinally the RFC hasnt a decision function so we calculate the probs of to be predicti in each class and take the value of the positive one
forest_clf = RandomForestClassifier(n_estimators=10, random_state=42)
y_probas_forest = cross_val_predict(forest_clf, X_train, y_train_5, cv=3, method="predict_proba")

# score = proba of positive class
y_scores_forest = y_probas_forest[:, 1] 
fpr_forest, tpr_forest, thresholds_forest = roc_curve(y_train_5,y_scores_forest)

#roc_curve_comparison_plot(fpr, tpr,fpr_forest,tpr_forest)
print("Random Forest AUC = ", roc_auc_score(y_train_5, y_scores_forest))

y_train_pred_forest = cross_val_predict(forest_clf, X_train, y_train_5, cv=3)

print("Random Forest Precision = ",precision_score(y_train_5, y_train_pred_forest))
print("Random Forest Recall = ",recall_score(y_train_5, y_train_pred_forest))

