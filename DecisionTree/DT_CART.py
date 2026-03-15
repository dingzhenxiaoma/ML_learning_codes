import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor  # 回归决策树
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

def DT_CART():
    '''
    | x   | 1    | 2   | 3    | 4   | 5   | 6    | 7   | 8   | 9   | 10   |
    | --- | ---- | --- | ---- | --- | --- | ---- | --- | --- | --- | ---- |
    | y   | 5.56 | 5.7 | 5.91 | 6.4 | 6.8 | 7.05 | 8.9 | 8.7 | 9   | 9.05 |
    '''
    x=np.array([1,2,3,4,5,6,7,8,9,10]).reshape(-1,1)
    y=np.array([5.56,5.7,5.91,6.4,6.8,7.05,8.9,8.7,9,9.05])
    model1=DecisionTreeRegressor(max_depth=1)
    model2=DecisionTreeRegressor(max_depth=3)
    model3=LinearRegression()
    model1.fit(x,y)
    model2.fit(x,y)
    model3.fit(x,y)

    # 模型预测
    x_test = np.arange(0.0, 10.0, 0.01).reshape(-1, 1)
    y_pred1=model1.predict(x_test)
    y_pred2=model2.predict(x_test)
    y_pred3=model3.predict(x_test)

    # 结果可视化
    plt.figure(figsize=(10,6))
    plt.scatter(x,y,label='data')
    plt.plot(x_test,y_pred1,label='max_depth=1')
    plt.plot(x_test,y_pred2,label='max_depth=3')
    plt.plot(x_test,y_pred3,label='linear regression')
    plt.legend()
    plt.savefig("DecisionTree\\data\\cart.png")
    plt.show()
    
if __name__ == '__main__':
    DT_CART()