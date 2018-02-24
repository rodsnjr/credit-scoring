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
FEATURE_SET = os.path.join(DIR, 'dataset', 'training-features.csv')
FEATURE_TEST_SET = os.path.join(DIR, 'dataset', 'test-features.csv')
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
    df = df.rename(columns = {'Unnamed: 0': 'Id'})
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

def load_test_feature_set():
    df = pd.read_csv(FEATURE_TEST_SET)
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