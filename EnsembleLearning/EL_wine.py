import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier     # 集成学习
from sklearn.metrics import accuracy_score

def EL_wine():
    # 读取数据
    data = pd.read_csv("EnsembleLearning\\data\\wine0501.csv")
    # 从标签列中过滤掉1类别
    data = data[data['Class label'] != 1]
    x=data[['Alcohol','Hue']]
    y=data['Class label']
    le=LabelEncoder()
    y=le.fit_transform(y)
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=23,stratify=y)
    
    # 1.单一决策树
    model1=DecisionTreeClassifier(max_depth=3)
    model1.fit(x_train,y_train)
    # 模型评估
    print("准确率：",model1.score(x_test,y_test))

    # 2.AdaBoost分类器->CART树，200棵
    model2=AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=3),n_estimators=200,learning_rate=0.1)
    model2.fit(x_train,y_train)
    # 模型评估
    print("准确率：",model2.score(x_test,y_test))

if __name__ == '__main__':
    EL_wine()