import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression

def entrenar_modelo(df, target_col):
    X = df.drop(columns=[target_col])
    y = df[target_col]

    model = LogisticRegression()
    model.fit(X, y)

    return model
