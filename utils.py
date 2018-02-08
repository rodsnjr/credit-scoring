"""

    This script contains utilities to load, plot, and helpers to the main notebooks of the project.

"""
import os
import pandas as pd
import numpy as np
from time import time
from sklearn.cross_validation import train_test_split
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score, auc, roc_curve

DIR = os.path.dirname(os.path.realpath(__file__))
SAMPLE_SET = os.path.join(DIR, 'dataset', 'sampleEntry.csv')
TEST_SET = os.path.join(DIR, 'dataset', 'cs-test.csv')
TRAIN_SET = os.path.join(DIR, 'dataset', 'cs-training.csv')
FEATURE_SET = os.path.join(DIR, 'dataset', 'features.csv')
DICTIONARY = os.path.join(DIR, 'dataset', 'Data Dictionary.xls')

def clean_columns(data):
    cleanCol = []
    for i in range(len(data.columns)):
        cleanCol.append(data.columns[i].replace('-', '').replace('.', ''))
    data.columns = cleanCol

def load_dictionary():
    df = pd.read_excel(DICTIONARY)
    return df

def load_test_set():
    df = pd.read_csv(TEST_SET, sep=',')
    df = df.drop('Unnamed: 0', axis = 1)
    clean_columns(df)
    return df

def load_training_set():
    df = pd.read_csv(TRAIN_SET, sep=',')
    df = df.drop('Unnamed: 0', axis = 1)
    clean_columns(df)
    return df

def load_features_set():
    df = pd.read_csv(FEATURE_SET)
    df = df.drop('Unnamed: 0', axis = 1)
    df = df.drop('X', axis = 1)
    clean_columns(df)
    return df

def load_sample():
    return pd.read_csv(SAMPLE_SET)

def removeSpecificAndPutMedian(data, first = 98, second = 96):
    New = []
    med = data.median()
    for val in data:
        if ((val == first) | (val == second)):
            New.append(med)
        else:
            New.append(val)
            
    return New

def basic_exploration(df):
    y = df['SeriousDlqin2yrs']
    yes = y.loc[y == 0].shape[0]
    no = y.loc[y == 1].shape[0]

    print('Total Number of persons in the set %s' % df.shape[0])
    print('Total Number number of persons with delinquency %s' % yes)
    print('Total Number number of persons without delinquency %s' % no)


def mad_based_outlier(points, thresh=3.5):
    if len(points.shape) == 1:
        points = points[:,None]
    median = np.median(points, axis=0)
    diff = np.sum((points - median)**2, axis=-1)
    diff = np.sqrt(diff)
    med_abs_deviation = np.median(diff)

    modified_z_score = 0.6745 * diff / med_abs_deviation

    return modified_z_score > thresh

def percentile_based_outlier(data, threshold=95):
    diff = (100 - threshold) / 2.0
    (minval, maxval) = np.percentile(data, [diff, 100 - diff])
    return ((data < minval) | (data > maxval))

def std_div(data, threshold=3):
    std = data.std()
    mean = data.mean()
    isOutlier = []
    for val in data:
        if val/std > threshold:
            isOutlier.append(True)
        else:
            isOutlier.append(False)
    return isOutlier

def outlier_ratio(data):
    functions = [percentile_based_outlier, mad_based_outlier, std_div, outlier_vote]
    outlierDict = {}
    for func in functions:
        funcResult = func(data)
        count = 0
        for val in funcResult:
            if val == True:
                count += 1 
        outlierDict[str(func)[10:].split()[0]] = [count, '{:.2f}%'.format((float(count)/len(data))*100)]
    
    return outlierDict

def outlier_vote(data):
    x = percentile_based_outlier(data)
    y = mad_based_outlier(data)
    z = std_div(data)
    temp = zip(data.index, x, y, z)
    final = []
    for i in range(len(temp)):
        if temp[i].count(False) >= 2:
            final.append(False)
        else:
            final.append(True)
    return final

def replace_outlier(data, method = outlier_vote, replace='median'):
    '''replace: median (auto)
                'minUpper' which is the upper bound of the outlier detection'''
    vote = outlierVote(data)
    x = pd.DataFrame(zip(data, vote), columns=['debt', 'outlier'])
    if replace == 'median':
        replace = x.debt.median()
    elif replace == 'minUpper':
        replace = min([val for (val, vote) in zip(data, vote) if vote == True])
        if replace < data.mean():
            return 'There are outliers lower than the sample mean'
    debtNew = []
    for i in range(x.shape[0]):
        if x.iloc[i][1] == True:
            debtNew.append(replace)
        else:
            debtNew.append(x.iloc[i][0])
    
    return debtNew

def describe_ages(data, min_age=16, max_age=30):
    ages = {}
    for i in range(min_age, max_age):
        ages[str(i)] = len(data[data.age < i])
    
    frame = pd.Series(ages)

    return frame

def percentage_missin(dataset):
    """this function will return the percentage of missing values in a dataset """
    if isinstance(dataset,pd.DataFrame):
        adict={} #a dictionary conatin keys columns names and values percentage of missin value in the columns
        for col in dataset.columns:
            adict[col]=(np.count_nonzero(dataset[col].isnull())*100)/len(dataset[col])
        return pd.DataFrame(adict,index=['% of missing'],columns=adict.keys())
    else:
        raise TypeError("can only be used with panda dataframe")

def split_dataset(x, y, portion=0.33):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=portion, random_state=42)

    print('Dataset splitted from: ')
    print('\tx=%s, y=%s' % (x.shape, y.shape))
    print('To: \nTrain Set')
    print('\tx=%s, y=%s' % (x_train.shape, y_train.shape))
    print('Test Set')
    print('\tx=%s, y=%s' % (x_test.shape, y_test.shape))

    return x_train, x_test, y_train, y_test

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