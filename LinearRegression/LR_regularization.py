import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error  # 计算均方误差
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso, Ridge
# L1正则化
def dm01_overfitting_L1():
    # 生成数据
    np.random.seed(23)
    x_1 = np.random.uniform(-3,3,size=100)
    y = 0.5*x_1**2+x_1+2+np.random.normal(0,1,size=100)

    x_1_t=x_1.reshape(-1,1)
    X=np.hstack([x_1_t,x_1_t**2,x_1_t**3,x_1_t**4,x_1_t**5,x_1_t**6,x_1_t**7,x_1_t**8,x_1_t**9,x_1_t**10])# 特征工程，添加二次项到十次项

    # 线性回归模型
    # model=LinearRegression()
    # L1正则化
    model=Lasso(alpha=0.1)
    model.fit(X,y)
    y_pred=model.predict(X)
    mse=mean_squared_error(y,y_pred)
    print("均方误差：",mse)
    # 可视化
    plt.scatter(x_1,y,color='blue',label='Data')
    plt.plot(np.sort(x_1),y_pred[np.argsort(x_1)],color='red',label='Linear Regression')
    plt.legend()
    plt.show()

# L2正则化
def dm02_overfitting_L2():
    # 生成数据
    np.random.seed(23)
    x_1 = np.random.uniform(-3,3,size=100)
    y = 0.5*x_1**2+x_1+2+np.random.normal(0,1,size=100)

    x_1_t=x_1.reshape(-1,1)
    X=np.hstack([x_1_t,x_1_t**2,x_1_t**3,x_1_t**4,x_1_t**5,x_1_t**6,x_1_t**7,x_1_t**8,x_1_t**9,x_1_t**10])# 特征工程，添加二次项到十次项

    # 线性回归模型
    #model=LinearRegression()
    # L2正则化
    model=Ridge(alpha=1.0)
    model.fit(X,y)
    y_pred=model.predict(X)
    mse=mean_squared_error(y,y_pred)
    print("均方误差：",mse)
    # 可视化
    plt.scatter(x_1,y,color='blue',label='Data')
    plt.plot(np.sort(x_1),y_pred[np.argsort(x_1)],color='red',label='Linear Regression')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    #dm01_overfitting_L1()
    dm02_overfitting_L2()