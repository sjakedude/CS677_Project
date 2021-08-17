"""
Jake Stephens
Class: CS 677 - Spring 2
Date: 8/22/2021
Project
Analyzing abalone data to find the best classifier to predict the number of rings.
"""

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
from sklearn import metrics


accuracy_table = pd.DataFrame(
        {
            "k": [],
            "tp": [],
            "fp": [],
            "tn": [],
            "fn": [],
            "accuracy": [],
            "tpr": [],
            "tnr": [],
        }
    )

def knn(x, y, k):
    
    # Splitting 50:50
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.5, random_state=1, shuffle=True
    )

    # Running KNN
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(x_train, y_train)
    y_pred = knn.predict(x_test)
    accuracy = metrics.accuracy_score(y_pred, y_test)

    index = 0
    tp = 0
    fp = 0
    tn = 0
    fn = 0

    for item in y_pred:
        if item == "T":
            if item == y_test[index]:
                tp += 1
            else:
                fp += 1
        else:
            if item == y_test[index]:
                tn += 1
            else:
                fn += 1
        index += 1
    tpr = tp / (tp + fn)
    tnr = tn / (tn + fp)

    accuracy_table = pd.DataFrame(
        {
            "tp": [tp],
            "fp": fp,
            "tn": [tn],
            "fn": [fn],
            "accuracy": [accuracy],
            "tpr": [tpr],
            "tnr": [tnr],
        }
    )

    print(accuracy_table)
    return accuracy
