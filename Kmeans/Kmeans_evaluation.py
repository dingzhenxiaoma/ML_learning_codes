import os
os.environ["OMP_NUM_THREADS"] = "4"
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.metrics import calinski_harabasz_score,silhouette_score

def kmeans_evaluation_sse():
    sse_list=[]
    x, y = make_blobs(n_samples=1000, n_features=2, 
                      centers=[[-1, -1], [0, 0], [1, 1], [2, 2]],
                      cluster_std=[0.4, 0.2, 0.2, 0.2], random_state=23)
    for k in range(1,100):
        kmeans=KMeans(n_clusters=k,max_iter=100,random_state=23)
        kmeans.fit(x)
        sse_list.append(kmeans.inertia_)
    plt.figure(figsize=(18, 8), dpi=100)
    plt.xticks(range(0, 100, 3), labels=range(0, 100, 3))
    plt.grid()
    plt.title('sse')
    plt.plot(range(1, 100), sse_list, 'or-')
    plt.show()

def kmeans_evaluation_sc():
    sc_list=[]
    x, y = make_blobs(n_samples=1000, n_features=2, 
                      centers=[[-1, -1], [0, 0], [1, 1], [2, 2]],
                      cluster_std=[0.4, 0.2, 0.2, 0.2], random_state=23)
    for k in range(2,100):
        kmeans=KMeans(n_clusters=k,max_iter=100,random_state=23)
        kmeans.fit(x)
        y_pred=kmeans.predict(x)
        sc_list.append(silhouette_score(x, y_pred))
    plt.figure(figsize=(18, 8), dpi=100)
    plt.xticks(range(0, 100, 3), labels=range(0, 100, 3))
    plt.grid()
    plt.title('sc')
    plt.plot(range(2, 100), sc_list, 'or-')
    plt.show()

def kmeans_evaluation_ch():
    ch_list=[]
    x, y = make_blobs(n_samples=1000, n_features=2, 
                      centers=[[-1, -1], [0, 0], [1, 1], [2, 2]],
                      cluster_std=[0.4, 0.2, 0.2, 0.2], random_state=23)
    for k in range(2,100):
        kmeans=KMeans(n_clusters=k,max_iter=100,random_state=23)
        kmeans.fit(x)
        y_pred=kmeans.predict(x)
        ch_list.append(calinski_harabasz_score(x, y_pred))
    plt.figure(figsize=(18, 8), dpi=100)
    plt.xticks(range(0, 100, 3), labels=range(0, 100, 3))
    plt.grid()
    plt.title('ch')
    plt.plot(range(2, 100), ch_list, 'or-')
    plt.show()

if __name__ == '__main__':
    #kmeans_evaluation_sse()
    #kmeans_evaluation_sc()
    kmeans_evaluation_ch()