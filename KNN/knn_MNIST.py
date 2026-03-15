import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import joblib
from collections import Counter

# 展示索引对应的图片
def show_digit(index):
    # 读取数据
    df=pd.read_csv("KNN\data\手写数字识别.csv")
    if index<0 or index>len(df)-1:
        print("索引超出范围")
        return
    # 提取索引对应的图片数据
    value=df.iloc[index,0]
    print(value)
    digit=df.iloc[index,1:].values.reshape(28,28)
    # 显示图片
    plt.imshow(digit,cmap=plt.cm.gray)
    plt.title(f"Digit: {df.iloc[index,-1]}")
    plt.axis("off")
    plt.show()

# 训练和保存模型
def train_model():
    # 读取数据
    df=pd.read_csv("KNN\data\手写数字识别.csv")
    # 划分数据集
    X=df.iloc[:,1:]/255.0
    y=df.iloc[:,0]
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=21,stratify=y)
    # 训练模型
    knn=KNeighborsClassifier(n_neighbors=3)
    knn.fit(X_train,y_train)
    # 评估模型
    y_pred=knn.predict(X_test)
    accuracy=accuracy_score(y_test,y_pred)
    print("KNN分类器的准确率：",accuracy)
    # 保存模型
    joblib.dump(knn,"KNN\model\knn_mnist_model.pkl")

# 加载模型
def load_model():
    img=plt.imread("KNN\data\demo.png")
    plt.imshow(img,cmap=plt.cm.gray)
    plt.axis("off")
    plt.show()
    knn=joblib.load("KNN\model\knn_mnist_model.pkl")
    y_pred=knn.predict(img.reshape(1,-1))
    print("预测结果：",y_pred[0])

if __name__ == '__main__':
    train_model()
    load_model()