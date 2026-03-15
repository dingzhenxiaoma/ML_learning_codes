import os
os.environ["OMP_NUM_THREADS"] = "1"
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score

def dm01_find_k():
    data=pd.read_csv("Kmeans\\data\\customers.csv")
    x=data.iloc[:,3:5]
    sse_list=[]
    sc_list=[]
    for k in range(2,100):
        kmeans=KMeans(n_clusters=k,max_iter=100,random_state=23)
        kmeans.fit(x)
        y_pred=kmeans.predict(x)
        sse_list.append(kmeans.inertia_)
        sc_list.append(silhouette_score(x, y_pred))
    
    plt.figure(figsize=(18, 8), dpi=100)
    plt.xticks(range(0, 100, 3), labels=range(0, 100, 3))
    plt.grid()
    plt.title('sse')
    plt.plot(range(2, 100), sse_list, 'or-')
    plt.show()
    
    plt.figure(figsize=(18, 8), dpi=100)
    plt.xticks(range(0, 100, 3), labels=range(0, 100, 3))
    plt.grid()
    plt.title('sc')
    plt.plot(range(2, 100), sc_list, 'or-')
    plt.show()

def dm02_train():
    dataset = pd.read_csv("Kmeans\\data\\customers.csv")
    X = dataset.iloc[:, [3, 4]]
    mykeans = KMeans(n_clusters=5)
    mykeans.fit(X)
    y_kmeans = mykeans.predict(X)
    plt.scatter(X.values[y_kmeans == 0, 0], X.values[y_kmeans == 0, 1], s=100, c='red', label='Standard')
    plt.scatter(X.values[y_kmeans == 1, 0], X.values[y_kmeans == 1, 1], s=100, c='blue', label='Traditional')
    plt.scatter(X.values[y_kmeans == 2, 0], X.values[y_kmeans == 2, 1], s=100, c='green', label='Normal')
    plt.scatter(X.values[y_kmeans == 3, 0], X.values[y_kmeans == 3, 1], s=100, c='cyan', label='Youth')
    plt.scatter(X.values[y_kmeans == 4, 0], X.values[y_kmeans == 4, 1], s=100, c='magenta', label='TA')
    plt.scatter(mykeans.cluster_centers_[:, 0], mykeans.cluster_centers_[:, 1], s=300, c='black', label='Centroids')
    
    plt.title('Clusters of customers')
    plt.xlabel('Annual Income (k$)')
    plt.ylabel('Spending Score (1-100)')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    # dm01_find_k()
    dm02_train()