import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.feature_selection import VarianceThreshold

def preparar_datos(df, target_col, grado=2, umbral_varianza=0.01):
    X = df.drop(columns=[target_col])
    y = df[target_col].to_numpy()

    pipeline = Pipeline(steps=[
        ("poly", PolynomialFeatures(degree=grado, include_bias=False)),
        ("var_filter", VarianceThreshold(threshold=umbral_varianza)),
        ("scaler", StandardScaler())
    ])

    X_transformada = pipeline.fit_transform(X)

    return X_transformada, y
