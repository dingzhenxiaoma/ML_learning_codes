import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler        # 特征处理
from sklearn.model_selection import train_test_split    # 数据集划分
from sklearn.linear_model import LinearRegression       # 正规方程的回归模型
from sklearn.linear_model import SGDRegressor           # 梯度下降的回归模型
from sklearn.metrics import mean_squared_error, root_mean_squared_error, mean_absolute_error          # 均方误差评估
from sklearn.linear_model import Ridge, RidgeCV

def boston_regression_Boston():
    # 加载数据集
    data_url = "http://lib.stat.cmu.edu/datasets/boston"
    raw_df = pd.read_csv(data_url, sep="\\s+", skiprows=22, header=None)
    data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
    target = raw_df.values[1::2, 2]
    # 划分数据集
    x_train, x_test, y_train, y_test = train_test_split(data, target, test_size=0.2, random_state=42)
    # 特征处理
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)
    # 训练模型
    #model=LinearRegression() # 正规方程的回归模型
    model=SGDRegressor(fit_intercept=True,learning_rate='constant',eta0=0.01) # 梯度下降的回归模型
    model.fit(x_train,y_train)
    # 模型评估
    y_pred=model.predict(x_test)
    mse=mean_squared_error(y_test,y_pred)
    rmse=root_mean_squared_error(y_test,y_pred)
    mae=mean_absolute_error(y_test,y_pred)
    print("均方误差：",mse)
    print("均方根误差：",rmse)
    print("平均绝对误差：",mae)

if __name__ == '__main__':
    boston_regression_Boston()