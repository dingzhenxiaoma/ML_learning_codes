from sklearn.preprocessing import MinMaxScaler,StandardScaler
def knn_min_max_normalization(X_train,mi,mx):
    scaler=MinMaxScaler(feature_range=(0,1))
    x_train_new=scaler.fit_transform(X_train)
    print(x_train_new)

def knn_standardization(X_train):
    scaler=StandardScaler()
    x_train_new=scaler.fit_transform(X_train)
    print(x_train_new)

if __name__ == '__main__':
    x_train = [[90, 2, 10, 40],
            [60, 4, 15, 45],
            [75, 3, 13, 46]]
    knn_min_max_normalization(x_train,0,1)
    knn_standardization(x_train)