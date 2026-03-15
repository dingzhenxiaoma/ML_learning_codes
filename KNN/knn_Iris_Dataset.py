from sklearn.datasets import load_iris
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# 加载Iris数据集
def dm01_load_iris():
    iris_data=load_iris()
    print("数据集：",iris_data.keys())
    print("特征名：",iris_data.feature_names)
    print("目标名：",iris_data.target_names)
    print("数据：",iris_data.data[:5])
    print("目标：",iris_data.target[:5])
    # frame:数据集的框架
    # DESCR：数据集的描述
    # filename：数据集的文件名
    # data_module：数据集的模型(在哪个包下)

# 绘制数据集的散点图
def dm02_show_iris():
    iris_data=load_iris()
    df=pd.DataFrame(iris_data.data,columns=iris_data.feature_names)
    df["target"]=iris_data.target
    sns.lmplot(df,x='sepal length (cm)',y='sepal width (cm)',hue="target",fit_reg=False)
    plt.title("Iris Dataset Scatter Plot")
    plt.tight_layout()
    plt.show()

# 数据集划分
def dm03_split_iris():
    iris_data=load_iris()
    X_train,X_test,y_train,y_test=train_test_split(iris_data.data,iris_data.target,test_size=0.2,random_state=23)
    return X_train,X_test,y_train,y_test

# 特征预处理
def dm04_preprocess_iris(X_train,X_test):
    scaler=StandardScaler()
    X_train_scaled=scaler.fit_transform(X_train)
    X_test_scaled=scaler.transform(X_test)
    return X_train_scaled,X_test_scaled,scaler

# KNN分类器
def dm05_knn_iris_classification(X_train,y_train,n_neighbors=5):
    knn=KNeighborsClassifier(n_neighbors=n_neighbors)
    knn.fit(X_train,y_train)
    return knn

# 模型评估
def dm06_evaluate_knn(knn,X_test,y_test):
    y_pred=knn.predict(X_test)
    accuracy=accuracy_score(y_test,y_pred)
    print("KNN分类器的准确率：",accuracy)

if __name__ == '__main__':
    dm01_load_iris()
    dm02_show_iris()
    X_train,X_test,y_train,y_test=dm03_split_iris()
    X_train_scaled,X_test_scaled,scaler=dm04_preprocess_iris(X_train,X_test)
    knn=dm05_knn_iris_classification(X_train_scaled,y_train)
    dm06_evaluate_knn(knn,X_test_scaled,y_test)
    # 自定义数据集
    X=[[7.8,2.1,3.9,1.6]]
    X_scaled=scaler.transform(X)
    print("预测结果：",knn.predict(X_scaled))
    print("各分类的预测概率：",knn.predict_proba(X_scaled))
