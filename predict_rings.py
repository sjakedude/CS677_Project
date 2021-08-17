"""
Jake Stephens
Class: CS 677 - Spring 2
Date: 8/22/2021
Project
Analyzing abalone data to find the best classifier to predict the number of rings.
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np
import seaborn as sn
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import sys


def linear_regression(df, column):
    # Separating into x and y
    x = df[column]
    y = df["Rings"]

    # Splitting 50:50
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.5, random_state=1, shuffle=True
    )

    # Training and testing the model
    degree = 1
    weights = np.polyfit(x_train, y_train, degree)
    model = np.poly1d(weights)
    predicted = model(x_test)

    plt.plot(x_test, y_test, "s", x_test, predicted, "s")
    plt.title("Linear Regression for predicting # of Rings")
    plt.xlabel(column)
    plt.ylabel("Rings")
    #plt.show()

    # Calculate sum of residuals squared
    sum_of_errors_squared = ((y_test - predicted) * (y_test - predicted)).sum()

    return sum_of_errors_squared


def quadradic(df, column):

    # Group 4 (x=platlets, y=serium creatinine)
    # Separating into x and y
    x = df[column]
    y = df["Rings"]

    # Splitting 50:50
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.5, random_state=1, shuffle=True
    )

    degree = 2
    weights = np.polyfit(x_train, y_train, degree)
    model = np.poly1d(weights)
    predicted = model(x_test)

    plt.plot(x_test, y_test, "s", x_test, predicted, "s")
    plt.title("Quadradic for Predicting # of Rings")
    plt.xlabel(column)
    plt.ylabel("Rings")
    #plt.show()

    sum_of_errors_squared = ((y_test - predicted) * (y_test - predicted)).sum()

    return sum_of_errors_squared

def cubic(df, column):

    # Group 4 (x=platlets, y=serium creatinine)
    # Separating into x and y
    x = df[column]
    y = df["Rings"]

    # Splitting 50:50
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.5, random_state=1, shuffle=True
    )

    degree = 3
    weights = np.polyfit(x_train, y_train, degree)
    model = np.poly1d(weights)
    predicted = model(x_test)

    plt.plot(x_test, y_test, "s", x_test, predicted, "s")
    plt.title("Cubic for Predicting # of Rings")
    plt.xlabel(column)
    plt.ylabel("Rings")
    #plt.show()

    sum_of_errors_squared = ((y_test - predicted) * (y_test - predicted)).sum()

    return sum_of_errors_squared

