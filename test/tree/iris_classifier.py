from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from myml.tree import DecisionTreeClassifier as myDecisionTreeClassifier

# 加载鸢尾花数据集
iris = load_iris()
X = iris.data
y = iris.target

# 将数据集拆分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

# 创建决策树分类器对象并训练模型
tree = DecisionTreeClassifier(criterion='gini')
tree.fit(X_train, y_train)

# 对测试集进行预测
y_pred = tree.predict(X_test)

# 评估模型性能
accuracy = accuracy_score(y_test, y_pred)
print("准确率:", accuracy)

# 创建决策树分类器对象并训练模型
tree = myDecisionTreeClassifier(criterion='gain_ratio',max_depth=2)
tree.fit(X_train, y_train)

# 对测试集进行预测
y_pred = tree.predict(X_test)

# 评估模型性能
accuracy = accuracy_score(y_test, y_pred)
print("准确率:", accuracy)
