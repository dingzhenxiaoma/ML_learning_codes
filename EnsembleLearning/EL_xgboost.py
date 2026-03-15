import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold
from sklearn.utils import class_weight
from sklearn.model_selection import GridSearchCV

def dm01_data_split():
    data=pd.read_csv("EnsembleLearning\\data\\红酒品质分类.csv")
    x=data.iloc[:,:-1]
    y=data.iloc[:,-1]-3
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=23,stratify=y)
    pd.concat([x_train,y_train],axis=1).to_csv("EnsembleLearning\\data\\红酒品质分类_train.csv",index=False)
    pd.concat([x_test,y_test],axis=1).to_csv("EnsembleLearning\\data\\红酒品质分类_test.csv",index=False)

def dm02_model_train():
    data_train=pd.read_csv("EnsembleLearning\\data\\红酒品质分类_train.csv")
    x_train=data_train.iloc[:,:-1]
    y_train=data_train.iloc[:,-1]
    data_test=pd.read_csv("EnsembleLearning\\data\\红酒品质分类_test.csv")
    x_test=data_test.iloc[:,:-1]
    y_test=data_test.iloc[:,-1]
    model=xgb.XGBClassifier(
        max_depth=5,
        learning_rate=0.1,
        random_state=23,
        objective='multi:softmax'
    )
    class_weight.compute_sample_weight(y_train)
    model.fit(x_train,y_train)
    y_pred=model.predict(x_test)
    print("模型分类报告：")
    print(classification_report(y_test,y_pred))
    joblib.dump(model,"EnsembleLearning\\model\\红酒品质分类_xgb_model.pkl")

def dm03_model_predict():
    model=joblib.load("EnsembleLearning\\model\\红酒品质分类_xgb_model.pkl")
    data_test=pd.read_csv("EnsembleLearning\\data\\红酒品质分类_test.csv")
    data_train=pd.read_csv("EnsembleLearning\\data\\红酒品质分类_train.csv")
    x_train=data_train.iloc[:,:-1]
    y_train=data_train.iloc[:,-1]
    x_test=data_test.iloc[:,:-1]
    y_test=data_test.iloc[:,-1]
    
    param_dict={
        'max_depth':[2,5,7],
        'learning_rate':[0.2,0.3,1.3],
        'n_estimators':[30,50,100,150]
    }
    skf=StratifiedKFold(n_splits=5,shuffle=True,random_state=23)
    grid_search=GridSearchCV(model,param_dict,cv=skf)
    grid_search.fit(x_train,y_train)
    model=grid_search.best_estimator_
    y_pred=model.predict(x_test)
    print("模型分类报告：")
    print(classification_report(y_test,y_pred))

if __name__ == '__main__':
    #dm01_data_split()
    #dm02_model_train()
    dm03_model_predict()