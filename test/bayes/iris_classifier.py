# 导入必要的库
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB

from myml.bayes import NaiveBayesClassifier
from myml.bayes import GaussianBayesClassfier



if __name__ == "__main__":



    # 加载示例数据集
    iris = load_iris()
    X = iris.data
    y = iris.target

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.7, random_state=42)


    # 创建自己手写的高斯正态贝叶斯分类器，并进行训练
    classifier = GaussianBayesClassfier()
    classifier.fit(X_train, y_train)
    # 进行预测
    y_pred = classifier.predict(X_test)
    # 计算准确率
    accuracy = accuracy_score(y_test, y_pred)
    print("GaussianBayesClassfier(myml)(class_prob_eq) accuracy:", accuracy)

    # 创建自己手写的高斯正态贝叶斯分类器，并进行训练
    classifier = GaussianBayesClassfier(class_prob_eq=False)
    classifier.fit(X_train, y_train)
    # 进行预测
    y_pred = classifier.predict(X_test)
    # 计算准确率
    accuracy = accuracy_score(y_test, y_pred)
    print("GaussianBayesClassfier(myml) accuracy:", accuracy)

    # 创建sklearn朴素贝叶斯高斯分类器对象
    gnb = GaussianNB()
    # 使用训练数据拟合模型
    gnb.fit(X_train, y_train)
    # 使用模型进行预测
    y_pred = gnb.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("GaussianNB(sklearn) accuracy:", accuracy)

    # 创建自己的朴素贝叶斯高斯分类器对象
    gnb = NaiveBayesClassifier()
    gnb.fit(X_train, y_train)
    # 在测试集上进行预测
    y_pred = gnb.predict(X_test)
    # 计算准确率
    accuracy = accuracy_score(y_test, y_pred)
    print("NaiveBayesClassifier(myml) accuracy:", accuracy)