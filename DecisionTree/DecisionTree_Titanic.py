import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

def decision_tree_train():
    # 读取数据
    data=pd.read_csv("DecisionTree\\data\\train.csv")
    x=data[['Pclass','Sex','Age']]
    y=data['Survived']
    x=x.copy()
    x['Age']=x['Age'].fillna(x['Age'].mean())
    x=pd.get_dummies(x,columns=['Sex'])
    # 数据集划分
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=23)
    # 模型训练
    model=DecisionTreeClassifier()
    model.fit(x_train,y_train)
    # 模型评估
    y_pred=model.predict(x_test)
    print("模型分类报告：")
    print(classification_report(y_test,y_pred))

    # 绘制决策树
    plt.figure(figsize=(200,100))
    plot_tree(model,feature_names=x.columns,class_names=['Not Survived','Survived'],filled=True)
    plt.savefig("DecisionTree\\data\\tree.png")

if __name__ == '__main__':
    decision_tree_train()