import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import jieba
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB   # 多项分布朴素贝叶斯
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

def train_model():
    data=pd.read_csv("Bayesian\\data\\书籍评价.csv",encoding="gbk")
    data['labels']=np.where(data['评价']=="好评",1,0)
    comment_list=[','.join(jieba.lcut(line)) for line in data['内容']]
    with open("Bayesian\\data\\stopwords.txt",'r',encoding="utf-8") as f:
        stop_words=f.readlines()
    stop_words=[line.strip() for line in stop_words]
    stop_word=list(set(stop_words))
    vectorizer=CountVectorizer(stop_words=stop_word)
    X=vectorizer.fit_transform(comment_list).toarray()
    Y=data['labels'].values
    x_train=X[:10]
    x_test=X[10:]
    y_train=Y[:10]
    y_test=Y[10:]
    model=MultinomialNB()
    model.fit(x_train,y_train)
    y_pred=model.predict(x_test)
    print(y_pred)
    print("模型分类报告：")
    print(classification_report(y_test,y_pred))

if __name__ == '__main__':
    train_model()