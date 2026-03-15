from sklearn.neighbors import KNeighborsClassifier,KNeighborsRegressor

def knn_classification(X_train,y_train,n_neighbors=5):
    knn=KNeighborsClassifier(n_neighbors=n_neighbors)
    knn.fit(X_train,y_train)
    return knn

def knn_regression(X_train,y_train,n_neighbors=5):
    knn=KNeighborsRegressor(n_neighbors=n_neighbors)
    knn.fit(X_train,y_train)
    return knn

def test_knn_classification():
    x_train=[[39,0,31],
            [3,2,65],
            [2,3,55],
            [9,38,2],
            [8,34,17],
            [5,2,57],
            [39,0,31],
            [21,17,5],
            [45,2,9]]
    y_train=["喜剧片",
            "动作片",
            "动作片",
            "爱情片",
            "爱情片",
            "动作片",
            "喜剧片",
            "喜剧片",
            "喜剧片"]
    x_test=[[23,3,17]]
    knn=knn_classification(x_train,y_train)
    y_pred=knn.predict(x_test)
    print(y_pred)

def test_knn_regression():
    x_train=[[0,0,1],[1,1,0],[3,10,10],[4,11,12]]
    y_train=[0.1,0.2,0.3,0.4]
    x_test=[[3,11,10]]
    knn=knn_regression(x_train,y_train,n_neighbors=2)
    y_pred=knn.predict(x_test)
    print(y_pred)

if __name__ == '__main__':
    test_knn_classification()
    test_knn_regression()