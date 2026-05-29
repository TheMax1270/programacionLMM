import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.cluster import AgglomerativeClustering

def validar_con_kfold_estratificado(df, feature_cols, label_col, n_clusters, n_splits):
    X = df[feature_cols].to_numpy()
    y = df[label_col].to_numpy()

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    purezas = []

    for _, test_idx in skf.split(X, y):
        X_test = X[test_idx]
        y_test = y[test_idx]

        labels = AgglomerativeClustering(n_clusters=n_clusters).fit_predict(X_test)

        aciertos = 0
        for c in np.unique(labels):
            clases_cluster = y_test[labels == c].astype(int)
            aciertos += np.bincount(clases_cluster).max()

        pureza = aciertos / len(y_test)
        purezas.append(pureza)

    return {
        "pureza_por_fold": purezas,
        "pureza_media": round(float(np.mean(purezas)), 4),
        "pureza_std": round(float(np.std(purezas)), 4)
    }
