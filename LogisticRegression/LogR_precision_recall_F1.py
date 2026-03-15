import pandas as pd
from sklearn.metrics import confusion_matrix,precision_score,recall_score,f1_score

def logR_precision_recall_F1():
    y_train=[0,0,0,0,0,0,1,1,1,1]
    y_pred_A=[0,0,0,1,1,1,1,1,1,1] # 模型A的预测结果: 正例预测对了3个，反例预测对了4个
    y_pred_B=[0,0,0,0,0,0,1,0,0,0] # 模型B的预测结果：正例预测对了6个，反例预测对了1个

    label=[0,1]
    df_label=['正例','反例']
    cm_A=confusion_matrix(y_train,y_pred_A)
    cm_B=confusion_matrix(y_train,y_pred_B)
    df_A=pd.DataFrame(cm_A,index=df_label,columns=df_label)
    df_B=pd.DataFrame(cm_B,index=df_label,columns=df_label)

    print("模型A的混淆矩阵：")
    print(df_A)
    print("模型B的混淆矩阵：")
    print(df_B)

    precision_A=precision_score(y_train,y_pred_A,pos_label=0)
    precision_B=precision_score(y_train,y_pred_B,pos_label=0)
    recall_A=recall_score(y_train,y_pred_A,pos_label=0)
    recall_B=recall_score(y_train,y_pred_B,pos_label=0)
    f1_A=f1_score(y_train,y_pred_A,pos_label=0)
    f1_B=f1_score(y_train,y_pred_B,pos_label=0)
    print("模型A的精确率：",precision_A)
    print("模型B的精确率：",precision_B)
    print("模型A的召回率：",recall_A)
    print("模型B的召回率：",recall_B)
    print("模型A的F1值：",f1_A)
    print("模型B的F1值：",f1_B)

if __name__ == '__main__':
    logR_precision_recall_F1()