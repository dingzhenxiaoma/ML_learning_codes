# 机器学习算法学习笔记
黑马程序员机器学习课程学习代码

本项目是一个机器学习算法的实践学习仓库，涵盖了从基础到进阶的经典 ML 算法，每个算法均配有完整的代码示例和真实/模拟数据集。

## 项目结构

```
ML_learning/
├── KNN/                    # K 近邻算法
├── LinearRegression/       # 线性回归
├── LogisticRegression/     # 逻辑回归
├── DecisionTree/           # 决策树
├── EnsembleLearning/       # 集成学习
├── Bayesian/               # 朴素贝叶斯
└── Kmeans/                 # K-Means 聚类
```

## 算法目录

### KNN（K 近邻）

| 文件 | 内容 |
|------|------|
| `knn.py` | KNN 分类与回归基础，自定义电影数据集 |
| `knn_preprocess.py` | 数据预处理：MinMaxScaler 归一化与 StandardScaler 标准化 |
| `knn_Iris_Dataset.py` | 鸢尾花数据集完整流程：加载、可视化、预处理、训练、评估 |
| `knn_GridSearchCV.py` | GridSearchCV 超参数调优与交叉验证 |
| `knn_MNIST.py` | MNIST 手写数字识别，模型持久化（joblib） |

### LinearRegression（线性回归）

| 文件 | 内容 |
|------|------|
| `LinearRegression.py` | 简单线性回归基础（身高-体重预测） |
| `LR_Boston.py` | 波士顿房价预测：正规方程 vs SGD，MSE/RMSE/MAE 评估 |
| `LR_fitting.py` | 欠拟合、正常拟合、过拟合的对比演示 |
| `LR_regularization.py` | L1 正则化（Lasso）与 L2 正则化（Ridge） |

### LogisticRegression（逻辑回归）

| 文件 | 内容 |
|------|------|
| `LogisticRegression_CancerPrediction.py` | 乳腺癌二分类预测，缺失值处理 |
| `LogR_precision_recall_F1.py` | 混淆矩阵、精确率、召回率、F1 分数 |
| `LogR_Telecom_customer_churn_pred.py` | 电信客户流失预测，分类变量编码 |

### DecisionTree（决策树）

| 文件 | 内容 |
|------|------|
| `DecisionTree_Titanic.py` | 泰坦尼克生存预测，决策树可视化 |
| `DT_CART.py` | CART 回归树，不同深度对比，与线性回归的比较 |

### EnsembleLearning（集成学习）

| 文件 | 内容 |
|------|------|
| `EnsembleLearning_RandomForest.py` | 随机森林 vs 单棵决策树，GridSearchCV 调参 |
| `EL_wine.py` | AdaBoost 自适应提升，葡萄酒数据集 |
| `EL_GBDT.py` | GBDT 梯度提升决策树，泰坦尼克数据集 |
| `EL_xgboost.py` | XGBoost 多分类，红酒品质预测，StratifiedKFold 交叉验证 |

### Bayesian（朴素贝叶斯）

| 文件 | 内容 |
|------|------|
| `Bayesian.py` | 多项式朴素贝叶斯，中文书评情感分类（jieba 分词 + CountVectorizer） |

### Kmeans（K-Means 聚类）

| 文件 | 内容 |
|------|------|
| `Kmeans.py` | K-Means 基础与聚类可视化 |
| `Kmeans_evaluation.py` | 聚类评估：SSE 肘部法、轮廓系数、Calinski-Harabasz 指数 |
| `Kmeans_customer.py` | 客户分群实战（收入-消费得分） |

## 技术栈

- Python 3
- scikit-learn
- NumPy / Pandas
- Matplotlib
- jieba（中文分词）
- XGBoost

## 涵盖知识点

- 数据预处理（归一化、标准化、缺失值处理、分类编码）
- 模型训练与评估（准确率、精确率、召回率、F1、MSE、RMSE）
- 超参数调优（GridSearchCV、交叉验证）
- 过拟合与正则化（L1/L2）
- 模型持久化（joblib）
- 聚类评估方法（SSE、轮廓系数、CH 指数）
