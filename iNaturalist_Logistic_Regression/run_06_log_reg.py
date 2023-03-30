import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

def train_logreg(df):
    x = df.iloc[:, :-1]
    y = df['label']
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.25)
    
    logreg = LogisticRegression(random_state=16)
    logreg.fit(X_train, y_train)
    y_pred = logreg.predict(X_test)
    
    fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred)
    auc = metrics.auc(fpr, tpr)
    
    return y_pred, auc