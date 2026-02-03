import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.pipeline import Pipeline
from dotenv import load_dotenv
import os
load_dotenv("/home/jpablo/BI/BI_Demo/.env")


def obtener_datos():
    df = pd.read_csv("https://raw.githubusercontent.com/JPMtz65/BI_Demo/refs/heads/main/bank_transactions_data_2.csv")

    pipeline = Pipeline([
        ('power', StandardScaler()),
        ('scaler', StandardScaler())
    ])

    X_transformed = pipeline.fit_transform(
        df[['TransactionAmount','CustomerAge']]
    )
    return X_transformed, df

def knmeans(X_transformed,df):
    kmeans_3 = KMeans(n_clusters=3)

    labels_3 = kmeans_3.fit_predict(X_transformed)
    centers_3 = kmeans_3.cluster_centers_

    # Distancia euclidiana al centroide asignado
    distances = np.linalg.norm(
        X_transformed - centers_3[labels_3],
        axis=1
    )

    df_kmean = df.copy()
    df_kmean['distance_to_centroid'] = distances
    threshold_995 = np.percentile(distances, 99.5)

    df_kmean['is_fraud'] = df_kmean['distance_to_centroid'] >= threshold_995
    df_kmean['is_fraud'].value_counts(normalize=True)

    return df_kmean

def dbscan(X_transformed, df):
    k = 4  # igual a min_samples, 2 x la dim
    eps = 0.18  # Gráfica de arriba

    dbscan = DBSCAN(
        eps=eps,
        min_samples=k
    )

    db_labels = dbscan.fit_predict(X_transformed)
    df_dbscan = df.copy()

    df_dbscan['dbscan_cluster'] = db_labels
    df_dbscan['dbscan_cluster'].value_counts()

    return df_dbscan

def subir_tabla(df, nombre):
    username = os.getenv("POSTGRES_USER")
    password = os.getenv("POSTGRES_PASSWORD")
    db_url = f"postgresql://{username}:{password}@localhost:5432/bi_demo"
    engine = create_engine(db_url)

    df.to_sql(nombre, engine, if_exists="replace", index=False)



if __name__ == "__main__":
    X_transformed, df = obtener_datos()
    df_kmean = knmeans(X_transformed, df.copy())
    df_dbscan = dbscan(X_transformed, df.copy())

    subir_tabla(df_kmean, "df_kmean")
    subir_tabla(df_dbscan,"df_dbscan")
