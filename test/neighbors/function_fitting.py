import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from myml.neighbors import KnnRegressor


# 定义多元函数：f(x1, x2) = x1^2 + x2^2
def multivariate_function(x1, x2):
    return x1**2 + x2**2

# 生成样本数据
np.random.seed(42)
n_samples = 10000
X1 = np.random.uniform(-5,5,n_samples)
X2 = np.random.uniform(-5,5,n_samples)
y = multivariate_function(X1, X2)

# 将X1、X2合并为特征矩阵X
X = np.column_stack((X1, X2))

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 创建并训练决策树回归器
dt_regressor = KnnRegressor(k=1)
dt_regressor.fit(X_train, y_train)

# 预测测试集
y_pred = dt_regressor.predict(X_test)

# 计算均方误差
mse = mean_squared_error(y_test, y_pred)
print("均方误差(MSE)：", mse)

# 绘制原函数的图像
fig = plt.figure(figsize=(10, 5))
ax = fig.add_subplot(121, projection='3d')
x1 = np.linspace(-5, 5, 100)
x2 = np.linspace(-5, 5, 100)
X1, X2 = np.meshgrid(x1, x2)
Y = multivariate_function(X1, X2)
ax.plot_surface(X1, X2, Y, cmap='viridis')
ax.set_xlabel('X1')
ax.set_ylabel('X2')
ax.set_zlabel('f(X1, X2)')
ax.set_title('Original Function')

# 绘制拟合后的图像
ax = fig.add_subplot(122, projection='3d')
Y_pred = dt_regressor.predict(np.column_stack((X1.ravel(), X2.ravel())))
Y_pred = Y_pred.reshape(X1.shape)
ax.plot_surface(X1, X2, Y_pred, cmap='viridis')
ax.set_xlabel('X1')
ax.set_ylabel('X2')
ax.set_zlabel('f(X1, X2)')
ax.set_title('Fitted Function')

plt.tight_layout()
plt.show()
