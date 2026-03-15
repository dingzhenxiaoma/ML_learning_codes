from sklearn.linear_model import LinearRegression

def linear_regression():
    x_train=[[160],[166],[172],[174],[180]]
    y_train=[56.3,60.6,65.1,68.5,75]
    # 训练模型
    model=LinearRegression()
    model.fit(x_train,y_train)
    # 模型评估
    x_test=[[176]]
    y_pred=model.predict(x_test)
    print("预测结果：",y_pred[0])

if __name__ == '__main__':
    linear_regression()