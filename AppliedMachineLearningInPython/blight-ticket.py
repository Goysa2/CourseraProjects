## Assignment 4 - Understanding and Predicting Property Maintenance Fines

# This assignment is based on a data challenge from the Michigan  Data
# Science Team([MDST](http: // midas.umich.edu / mdst /)).

# The Michigan Data Science Team([MDST](http: // midas.umich.edu / mdst /)) and the Michigan Student Symposium for
# Interdisciplinary Statistical Sciences ([MSSISS](https://sites.lsa.umich.edu / mssiss /)) have partnered with the City
# of Detroit to help solve one of the most pressing problems facing Detroit -
# blight.[Blight violations](http://www.detroitmi.gov / How - Do - I / Report / Blight - Complaint - FAQs) are issued
# by the city to individuals who allow their properties to remain in a deteriorated condition. Every year, the city of
# Detroit issues millions of dollars in fines to residents and every year, many of these fines remain unpaid. Enforcing
# unpaid blight fines is a costly and tedious process, so the city wants to know: how can we increase blight ticket
# compliance?

# The first step in answering this question is understanding when and why a resident might fail to comply with a blight
# ticket.This is where predictive modeling comes in.For this assignment, your task is to predict whether a given blight
# ticket will be paid on time.

# All data for this assignment has been provided to us through the[Detroit Open Data Portal](https://data.detroitmi.gov /).**
# Only the data already included in your Coursera directory can be used for training the model for this assignment.
# ** Nonetheless, we encourage you to look into data from other Detroit datasets to help inform feature creation and
# model selection.We recommend taking a look at the following related datasets:

# *[Building Permits](https: // data.detroitmi.gov / Property - Parcels / Building - Permits / xw2a - a7tf)
# *[Trades Permits](https: // data.detroitmi.gov / Property - Parcels / Trades - Permits / 635
# b - dsgv)
# *[Improve Detroit: Submitted Issues](https: // data.detroitmi.gov / Government / Improve-Detroit-Submitted-Issues / fwz3-w3yn)
# *[DPD: Citizen Complaints](https: // data.detroitmi.gov / Public-Safety / DPD-Citizen-Complaints-2016 / kahe-efs3)
# *[Parcel Map](https: // data.detroitmi.gov / Property - Parcels / Parcel - Map / fxkw - udwf)

# We provide you with two data files for use in training and validating your models: train.csv and test.csv.Each row in
# these two files corresponds to a single blight ticket, and includes information about when, why, and to whom each
# ticket was issued. The target variable is compliance, which is True if the ticket was paid early, on time, or within
# one month of the hearing data, False if the ticket was paid after the hearing date or not at all, and Null if the
# found not responsible. Compliance, as well as a handful of other variables that will not be available at test - time,
# are only included in train.csv.

# Note: All tickets where the violators werefound not responsible are not considered during evaluation.They are included
# in the training set as an additional source of data for visualization, and to enable unsupervised and semi-supervised
# approaches.However, they are not included in the test set.

# ** File descriptions ** (Use only this data for training your model!)

# readonly / train.csv - the training set(all tickets issued 2004 - 2011)
# readonly / test.csv - the test set (all tickets issued 2012 - 2016)
# readonly / addresses.csv & readonly / latlons.csv - mapping from ticket id to addresses, and from addresses to lat / lon
# coordinates. Note: misspelled addresses may be incorrectly geolocated.

# ** Data fields **

# train.csv & test.csv

# ticket_id - unique  identifier for tickets
# agency_name - Agency that issued the ticket
# inspector_name - Name of inspector that issued the ticket
# violator_name - Name of the person / organization that the ticket was issued to
# violation_street_number, violation_street_name, violation_zip_code - Address where the violation occurred
# mailing_address_str_number, mailing_address_str_name, city, state, zip_code, non_us_str_code, country - Mailing  address of the violator
# ticket_issued_date - Date and time the ticket was issued
# hearing_date - Date and time the violator 's hearing was scheduled
# violation_code, violation_description - Type of  violation
# disposition - Judgment and judgement type
# fine_amount - Violation fine amount, excluding fees
# admin_fee - $20 fee assigned to responsible judgments
# state_fee - $10 fee assigned to responsible judgments
# late_fee - 10 % fee assigned to responsible judgments
# discount_amount - discount applied, if any clean_up_cost - DPW clean - up or graffiti removal cost
# judgment_amount - Sum of all fines and fees grafitti_status - Flag for graffiti violations

# train.csv only
# payment_amount - Amount paid, if any payment_date - Date payment was made,if it was received
# payment_status - Current payment status as of Feb 1 2017
# balance_due - Fines and fees still owed collection_status - Flag for payments in collections
# compliance [target variable for prediction]
#     Null = Not responsible
#     0 = Responsible, non - compliant
#     1 = Responsible, compliant
# compliance_detail - More information on why each ticket was marked compliant or non - compliant


## Evaluation
# Your predictions will be given as the probability that the corresponding blight ticket will be paid on time.
# The evaluation metric for this assignment is the Area Under the ROC Curve (AUC).

# Your grade will be based on the AUC score computed for your classifier. A model which with an AUROC of 0.7 passes this
# assignment, over 0.75 will recieve full points.

# For  this assignment, create a function that trains a model to predict blight ticket compliance in Detroit using `readonly / train.csv`.
# Using this model, return a series of length 61001 with the data being the probability that each corresponding ticket
# from `readonly / test.csv` will be paid, and the index being the ticket_id.

# Example:

# ticket_id
# 284932 0.531842
# 285362 0.401958
# 285361 0.105928
# 285338 0.018572
...
# 376499 0.208567
# 376500 0.818759
# 369851 0.018528
# Name: compliance, dtype: float32
#
### Hints

# Make sure your code is working before submitting it to the autograder.
# Print out your result to see whether there is anything weird(e.g., all probabilities are the same).
# Generally the total runtime should be less than 10 mins. You should NOT use Neural Network related classifiers (e.g.,
# MLPClassifier) in this  question.
# Try to avoid  global variables. If you have  other functions besides blight_model, you  should  move those functions
# inside the scope of blight_model.
# Refer to the pinned threads in Week 4 's discussion forum when there is something you could not figure it out.

import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyClassifier
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_curve, auc


def blight_model():
    ### Step 1
    # Figure out the data set
    # (first run it without adress or localization)

    ### Step 2
    # understand how to run a dummy classifier, predict_proba,etc..
    # learn how to run everything with predict proba and a dummy classifier

    ### Step 3
    # with a correct type of output and a dummy classifier, go back and fix things with a correct classifier
    # figure out how to do to know best features to use

    ### Step 1
    ## First try : no preprocessing : doesn't work because strings (object) can't be fitted
    train_df = pd.read_csv('train.csv', engine='python')
    test_df = pd.read_csv('test.csv', engine='python')

    ## Second try : some preprocessing
    ## Some preprocessing needs to be done on train set AND test set
    train_df = train_df.dropna(subset=['compliance'])  # avoid working with missing values
    train_df = train_df.dropna(axis=1)  # avoid working with missing values
    train_df = train_df.drop(['compliance_detail', 'country'], axis=1)  # avoid data leakage
    train_df = train_df.drop(['ticket_issued_date'], axis=1)  # for now no time series
    train_df = train_df.drop(['payment_amount', 'balance_due', 'payment_status'], axis=1)  # not in test set
    train_df = train_df.set_index('ticket_id')

    test_df = test_df.set_index('ticket_id')

    # transform categorical data in numerical data
    le = LabelEncoder()
    cat_features = ['agency_name', 'inspector_name', 'violation_street_name', 'city', 'violation_code',
                    'violation_description', 'disposition']
    for feat in cat_features:
        train_df[feat] = le.fit_transform(train_df[feat].astype('str'))
        if feat in test_df.columns:
            test_df[feat] = le.fit_transform(test_df[feat].astype('str'))

    ## Split data and labels
    X = train_df.drop('compliance', axis=1)
    y = train_df['compliance']

    X_t = test_df[X.columns]  # we want to have the same features ad

    X_train, X_test, y_train, y_test = train_test_split(X, y)
    clf = GradientBoostingClassifier().fit(X_train, y_train)

    return pd.Series(clf.predict_proba(X_t)[:, 1], index=X_t.index)


blight_model()