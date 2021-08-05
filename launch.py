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
from linear_regression import linear_regression
from nearest_neighbor import knn


# Reading in dataframe
df = pd.read_csv(
    "data/abalone.data",
    names=[
        "Sex",
        "Length",
        "Diameter",
        "Height",
        "Whole weight",
        "Shucked weight",
        "Viscera weight",
        "Shell weight",
        "Rings",
    ],
)


def main():

    # Using linear regression to find the best
    # column to predict the correct number of rings
    best_column = None
    lowest_sse = sys.float_info.max
    errors_by_column = {}
    for c in df.columns:
        if c not in ["Rings", "Sex"]:
            sse = linear_regression(df, c)
            errors_by_column[c] = [sse]
            if sse < lowest_sse:
                lowest_sse = sse
                best_column = c
    print("==========================================================")
    print("For predicting the number of rings using Linear Regression")
    print("==========================================================")
    print("Best Column: " + best_column)
    print("\nTable: (Sorted by best to worst from left to right)\n")
    errors_table = pd.DataFrame(
        {k: v for k, v in sorted(errors_by_column.items(), key=lambda item: item[1])}
    )
    print(errors_table)
    print("\n==========================================================")

    # Using Nearest Neighbor to find the best
    # column to predict the correct Sex
    neighbors = [3, 5, 7, 9, 11, 13]
    best_column = None
    best_accuracy = 0
    accuracy_by_k = {}
    for k in neighbors:
        accuracy = knn(df, k)
        accuracy_by_k[k] = [accuracy]
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_column = c
    print("==========================================================")
    print("For predicting the Sex using Nearest Neighbor")
    print("==========================================================")
    print("Best Column: " + best_column)
    print("\nTable: (Sorted by best to worst from left to right)\n")
    accuracy_by_k_table = pd.DataFrame(
        {
            k: v
            for k, v in sorted(
                accuracy_by_k.items(), key=lambda item: item[1], reverse=True
            )
        }
    )
    print(accuracy_by_k_table)
    print("\n==========================================================")


main()
