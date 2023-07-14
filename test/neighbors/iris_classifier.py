# 导入必要的库
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from myml.neighbors import KnnClassifier


if __name__ == "__main__":



    # 加载示例数据集
    iris = load_iris()
    X = iris.data
    y = iris.target

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.7, random_state=42)
    # 创建和训练分类器
    classifier = KNeighborsClassifier()
    classifier.fit(X_train, y_train)

    # 进行预测
    y_pred = classifier.predict(X_test)

    # 计算准确率
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)

    classifier = KnnClassifier()
    classifier.fit(X_train, y_train)

    # 进行预测
    y_pred = classifier.predict(X_test)

    # 计算准确率
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)