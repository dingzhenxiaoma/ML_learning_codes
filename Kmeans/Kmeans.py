from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.metrics import calinski_harabasz_score

def kmeans():
    X,Y= make_blobs(n_samples=1000,n_features=2,centers=[[-1,-1],[0,0],[1,1],[2,2]], 
                      cluster_std=[0.4,0.2,0.3,0.4], random_state=23)
    plt.scatter(X[:,0],X[:,1])
    plt.show()
    kmeans=KMeans(n_clusters=4,random_state=0)
    kmeans.fit(X)
    plt.scatter(X[:,0],X[:,1],c=kmeans.labels_)
    plt.show()

if __name__ == '__main__':
    kmeans()