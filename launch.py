"""
Jake Stephens
Class: CS 677 - Spring 2
Date: 8/22/2021
Project
Analyzing abalone data to find the best classifier to predict the number of rings.
"""

from predict_rings import quadradic
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
from predict_rings import linear_regression, quadradic, cubic
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

    # =============================================
    # Predicting: # of Rings
    # =============================================

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
    print("\nTable of SSE: (Sorted by best to worst from left to right)\n")
    errors_table = pd.DataFrame(
        {k: v for k, v in sorted(errors_by_column.items(), key=lambda item: item[1])}
    )
    print(errors_table)

    # Using quadratic to find the best column 
    # to predict the correct number of rings
    best_column = None
    lowest_sse = sys.float_info.max
    errors_by_column = {}
    for c in df.columns:
        if c not in ["Rings", "Sex"]:
            sse = quadradic(df, c)
            errors_by_column[c] = [sse]
            if sse < lowest_sse:
                lowest_sse = sse
                best_column = c
    print("==========================================================")
    print("For predicting the number of rings using Quatradic Method")
    print("==========================================================")
    print("Best Column: " + best_column)
    print("\nTable of SSE: (Sorted by best to worst from left to right)\n")
    errors_table = pd.DataFrame(
        {k: v for k, v in sorted(errors_by_column.items(), key=lambda item: item[1])}
    )
    print(errors_table)

    # Using cubic to find the best column 
    # to predict the correct number of rings
    best_column = None
    lowest_sse = sys.float_info.max
    errors_by_column = {}
    for c in df.columns:
        if c not in ["Rings", "Sex"]:
            sse = cubic(df, c)
            errors_by_column[c] = [sse]
            if sse < lowest_sse:
                lowest_sse = sse
                best_column = c
    print("==========================================================")
    print("For predicting the number of rings using Cubic Method")
    print("==========================================================")
    print("Best Column: " + best_column)
    print("\nTable of SSE: (Sorted by best to worst from left to right)\n")
    errors_table = pd.DataFrame(
        {k: v for k, v in sorted(errors_by_column.items(), key=lambda item: item[1])}
    )
    print(errors_table)
    print("\n\n")

    # =============================================
    # Predicting: Adult vs Child
    # =============================================

    # Now to create a new label called 'Adult'.
    # If the number of rings is 10 of greater, we will set
    # 'Adult' to T from True. If the number of rings is
    # less than 10, we will set F for False.
    # Adding Color column based on class
    def color_label(row):
        if row["Rings"] > 10:
            return "T"
        else:
            return "F"

    df["Adult"] = df.apply(lambda row: color_label(row), axis=1)

    # Separating into x and y
    y = df["Adult"].values.tolist()

    x = df.drop(["Adult", "Rings", "Sex"], axis=1)

    # Using Nearest Neighbor to find the best
    # column to predict the correct Sex
    neighbors = [3, 5, 7, 9, 11, 13]
    accuracy_by_k = {}
    for k in neighbors:
        accuracy = knn(x, y, k)
        accuracy_by_k[k] = [accuracy]
    print("==========================================================")
    print("For predicting Adult vs Child using Nearest Neighbor")
    print("==========================================================")
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
