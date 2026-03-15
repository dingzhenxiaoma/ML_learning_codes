import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def LogisticRegression_CancerPrediction():
    data=pd.read_csv('LogisticRegression\\data\\breast-cancer-wisconsin.csv')
    data.replace('?',np.nan,inplace=True)
    data.dropna(axis=0,inplace=True)
    X=data.iloc[:,1:-1]
    y=data.iloc[:,-1]
    # 数据标准化
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    # 划分数据集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # 训练模型
    model=LogisticRegression()
    model.fit(X_train,y_train)
    # 模型评估
    y_pred=model.predict(X_test)
    acc=accuracy_score(y_test,y_pred)
    print("准确率：",acc)

if __name__ == '__main__':
    LogisticRegression_CancerPrediction()