import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report

def train():
    # 读取数据
    data = pd.read_csv("EnsembleLearning\\data\\train.csv")
    x=data[['Pclass','Sex','Age']]
    y=data['Survived']
    x=x.copy()
    x['Age']=x['Age'].fillna(x['Age'].mean())
    x=pd.get_dummies(x,columns=['Sex'])
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=23)
    
    # 模型训练
    model1=DecisionTreeClassifier()
    model1.fit(x_train,y_train)
     # 模型评估
    print("准确率：",model1.score(x_test,y_test))

    # 模型训练
    model2=RandomForestClassifier()
    model2.fit(x_train,y_train)
     # 模型评估
    print("准确率：",model2.score(x_test,y_test))

    # 模型训练
    params={
        'n_estimators':[30,50,60,90,110],
        'max_depth':[3,5,7,9]
    }
    model3=RandomForestClassifier()
    grid_search=GridSearchCV(model3,params,cv=2)
    grid_search.fit(x_train,y_train)
     # 模型评估
    y_pred=grid_search.predict(x_test)
    print("准确率：",grid_search.score(x_test,y_test))

if __name__ == '__main__':
    train()
