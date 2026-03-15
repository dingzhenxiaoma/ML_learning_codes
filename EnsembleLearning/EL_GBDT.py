import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV

def EL_GBDT():
    # 读取数据
    data=pd.read_csv("EnsembleLearning\\data\\train.csv")
    x=data[['Pclass','Sex','Age']]
    y=data['Survived']
    x=x.copy()
    x['Age']=x['Age'].fillna(x['Age'].mean())
    x=pd.get_dummies(x,columns=['Sex'])
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=23)

    # 模型训练

    model1=DecisionTreeClassifier(max_depth=3)
    model1.fit(x_train,y_train)
    y_pred=model1.predict(x_test)
    print("模型分类报告：")
    print(classification_report(y_test,y_pred))

    model2=GradientBoostingClassifier()
    model2.fit(x_train,y_train)
    y_pred=model2.predict(x_test)
    print("模型分类报告：")
    print(classification_report(y_test,y_pred))

    param_dict={
        'n_estimators':[50,70,90,110],
        'learning_rate':[0.2,0.5,0.7],
        'max_depth':[3,6,8,9]
    }
    model3=GradientBoostingClassifier()
    grid_search=GridSearchCV(model3,param_dict,cv=2)
    grid_search.fit(x_train,y_train)
    model3=grid_search.best_estimator_
    y_pred=model3.predict(x_test)
    print("模型分类报告：")
    print(classification_report(y_test,y_pred))

if __name__ == '__main__':
    EL_GBDT()