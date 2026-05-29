import pandas as pd
import numpy as np
from sklearn.cluster import KMeans

def segmentar_y_calificar_clientes(dataset, features, target_score_parts):
    df = dataset.copy()

    X = df[features]

    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X)

    df["cluster"] = clusters
    df["score_valor"] = df[target_score_parts[0]] * df[target_score_parts[1]]

    centroides = kmeans.cluster_centers_

    idx_toneladas = features.index("toneladas_pedidas")

    cluster_b2b = int(np.argmax(centroides[:, idx_toneladas]))
    cluster_b2c = int(np.argmin(centroides[:, idx_toneladas]))

    top_b2b = df[df["cluster"] == cluster_b2b].loc[
        df[df["cluster"] == cluster_b2b]["score_valor"].idxmax(),
        "id_cliente"
    ]

    top_b2c = df[df["cluster"] == cluster_b2c].loc[
        df[df["cluster"] == cluster_b2c]["score_valor"].idxmax(),
        "id_cliente"
    ]

    return {
        "B2B_top_cliente": top_b2b,
        "B2C_top_cliente": top_b2c
    }
