import pandas as pd
import numpy as np

def segmentar_y_calificar_clientes(dataset, features, target_score_parts):
    df = dataset.copy()

    df["score_valor"] = df[target_score_parts[0]] * df[target_score_parts[1]]

    b2b_mask = df["id_cliente"].str.contains("CORP")
    b2c_mask = df["id_cliente"].str.contains("CONT")

    top_b2b = df[b2b_mask].loc[
        df[b2b_mask]["score_valor"].idxmax(),
        "id_cliente"
    ]

    top_b2c = df[b2c_mask].loc[
        df[b2c_mask]["score_valor"].idxmax(),
        "id_cliente"
    ]

    return {
        "B2B_top_cliente": top_b2b,
        "B2C_top_cliente": top_b2c
    }
