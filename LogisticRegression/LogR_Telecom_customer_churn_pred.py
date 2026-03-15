import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score,classification_report

def dm01_data_preprocess():
    # 读取数据
    df=pd.read_csv("LogisticRegression\\data\\churn.csv")
    df=pd.get_dummies(df,columns=['Churn','gender'])
    df.drop(columns=['Churn_No','gender_Male'],inplace=True)
    df.rename(columns={'Churn_Yes':'flag'},inplace=True)
    return df

def dm02_data_visualization(df):
    print(df.columns)
    '''
    Index(['Partner_att', 'Dependents_att', 'landline', 'internet_att',
       'internet_other', 'StreamingTV', 'StreamingMovies', 'Contract_Month',
       'Contract_1YR', 'PaymentBank', 'PaymentCreditcard', 'PaymentElectronic',
       'MonthlyCharges', 'TotalCharges', 'flag', 'gender_Female'],
      dtype='str')
    '''
    # 计数柱状图
    sns.countplot(x='Contract_Month',data=df,hue='flag')
    plt.show()

def dm03_train(df):
    # 特征提取
    x=df[['Contract_Month','internet_other','PaymentElectronic']]
    y=df['flag']
    # 数据集划分
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=23)
    # 模型训练
    model=LogisticRegression()
    model.fit(x_train,y_train)
    # 模型评估
    y_pred=model.predict(x_test)
    print("模型准确率：",accuracy_score(y_test,y_pred))
    print("模型精确率：",precision_score(y_test,y_pred))
    print("模型召回率：",recall_score(y_test,y_pred))
    print("模型F1值：",f1_score(y_test,y_pred))
    print("模型分类报告：")
    print(classification_report(y_test,y_pred))

if __name__ == '__main__':
    df=dm01_data_preprocess()
    #dm02_data_visualization(df)
    dm03_train(df)
