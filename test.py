import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, datasets

# 加載iris數據集
iris = datasets.load_iris()
X = iris.data[:, :2]
y = iris.target

# 將數據集劃分為訓練集和測試集
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# 設置SVM模型的參數
C = 1.0  # SVM正則化參數
svc = svm.SVC(kernel='linear', C=C).fit(x_train, y_train)  # 線性核SVM
rbf_svc = svm.SVC(kernel='rbf', gamma=0.7, C=C).fit(x_train, y_train)  # 径向基核SVM
poly_svc = svm.SVC(kernel='poly', degree=3, C=C).fit(x_train, y_train)  # 多項式核SVM
lin_svc = svm.LinearSVC(C=C).fit(x_train, y_train)  # 線性核SVM

# 可視化SVM模型的決策邊界和支持向量
def plot_boundary(model, ax=None):
    if ax is None:
        ax = plt.gca()
    # 獲取數據範圍
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    # 生成網格點
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                         np.arange(y_min, y_max, 0.02))
    # 預測網格點的類別
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    # 繪製決策邊界和支持向量
    ax.contourf(xx, yy, Z, alpha=0.2)
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired)

# 可視化SVM模型的決策邊界和支持向量
fig, sub = plt.subplots(2, 2, figsize=(10, 10))
plt.subplots_adjust(wspace=0.4, hspace=0.4)
models = (svc, rbf_svc, poly_svc, lin_svc)
titles = ('線性核SVM',
          '径向基核SVM',
          '多項式核SVM',
          '線性核SVM（LinearSVC）')

for model, title, ax in zip(models, titles, sub.flatten()):
    plot_boundary(model, ax)
    ax.set_title(title)

plt.show()

