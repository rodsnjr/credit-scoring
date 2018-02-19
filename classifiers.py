from time import time
from sklearn.cross_validation import train_test_split
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score, auc, roc_curve

from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB 
from sklearn.ensemble import RandomForestClassifier
from mlxtend.classifier import StackingClassifier
import numpy as np


def test_multiple_cv(classifiers, k, x, y, test_x, test_y):
    print('%s-fold cross validation:\n' % k)

    for clf in classifiers:
        scores = model_selection.cross_val_score(clf, X, y, 
                                                cv=k, scoring='roc_auc')
        print("ROC AUC: %0.2f (+/- %0.2f) [%s]" 
            % (scores.mean(), scores.std(), clf.__class__.__name__))

# From the Udacity Student Intervention Notebook
def train_classifier(clf, X_train, y_train):
    ''' Fits a classifier to the training data. '''
    
    # Start the clock, train the classifier, then stop the clock
    start = time()
    clf.fit(X_train, y_train)
    end = time()
    
    # Print the results
    print "Trained model in {:.4f} seconds".format(end - start)

    return clf


def predict_labels(clf, features, target):
    ''' Makes predictions using a fit classifier based on F1 score. '''
    
    # Start the clock, make predictions, then stop the clock
    start = time()
    y_pred = clf.predict(features)
    #y_proba = clf.predict_proba(features)
    end = time()
    
    # Print and return results
    print "Made predictions in {:.4f} seconds.".format(end - start)
    return f1_score(target, y_pred), roc_auc_score(target, y_pred)


def train_predict(clf, X_train, y_train, X_test, y_test):
    ''' Train and predict using a classifer based on F1 score. '''
    
    # Indicate the classifier and the training set size
    print "Training a {} using a training set size of {}. . .".format(clf.__class__.__name__, len(X_train))
    
    # Train the classifier
    clf = train_classifier(clf, X_train, y_train)
    
    f1_train, auc_train = predict_labels(clf, X_train, y_train)
    f1_test, auc_test = predict_labels(clf, X_test, y_test)
    # Print the results of prediction for both training and testing
    print "F1 score for training set: {:.4f}, and AUC {:.4f}"\
        .format(f1_train, auc_train)
    print "F1 score for test set: {:.4f} and AUC {:.4f}."\
        .format(f1_test, auc_test)
    
    return clf