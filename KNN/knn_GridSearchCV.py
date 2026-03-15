from sklearn.datasets import load_iris          # 加载鸢尾花测试集的.
from sklearn.model_selection import train_test_split, GridSearchCV    # 分割训练集和测试集的, 网格搜索的
from sklearn.preprocessing import StandardScaler        # 数据标准化的
from sklearn.neighbors import KNeighborsClassifier      # KNN算法 分类对象
from sklearn.metrics import accuracy_score              # 模型评估的, 计算模型预测的准确率

if __name__ == '__main__':
    iris_dataset=load_iris()
    x_train,x_test,y_train,y_test=train_test_split(iris_dataset.data,iris_dataset.target,test_size=0.2,random_state=22)
    scaler=StandardScaler()
    x_train_scaled=scaler.fit_transform(x_train)
    x_test_scaled=scaler.transform(x_test)
    knn=KNeighborsClassifier()
    param_grid={'n_neighbors':range(1, 11)}
    grid_search=GridSearchCV(knn,param_grid,cv=4)
    grid_search.fit(x_train_scaled,y_train)
    print("最佳参数：",grid_search.best_params_)
    print("最佳模型：",grid_search.best_estimator_)
    estimator=grid_search.best_estimator_
    y_pred=estimator.predict(x_test_scaled)
    accuracy=accuracy_score(y_test,y_pred)
    print("KNN分类器的准确率：",accuracy)