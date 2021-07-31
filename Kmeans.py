import numpy as np
from matplotlib import pyplot


class Kmeans(object):
    # k是分组数；tolerance‘中心点误差’；max_iter是迭代次数
    def __init__(self, k=2, tolerance=0.0001, max_iter=300):
        self.k = k
        self.tolerance = tolerance
        self.max_iter = max_iter

    def fit(self, data):
        self.centers = {}
        for i in range(self.k):
            self.centers[i] = data[i]   # 设置k个中心点，挑选输入数据点的前k个暂时作为中心点

        for i in range(self.max_iter):
            self.clf = {}   # 存放属于不同类别的数据点
            for i in range(self.k):   # 创建k个位置，用index代指某个类别
                self.clf[i] = []
            for feature in data:
                distances = []
                for center in self.centers:
                    distances.append(np.linalg.norm(feature - self.centers[center]))
                classification = distances.index(min(distances))    # 记录距离最小的类别
                self.clf[classification].append(feature)    # 将当前数据点分类至与其距离最小的类
            prev_centers = dict(self.centers)   # 一次分类完成，重新计算当前分类情况下每个类别的中心点
            for c in self.clf:
                self.centers[c] = np.average(self.clf[c], axis=0)

            # '中心点'是否在误差范围
            optimized = True
            for center in self.centers:
                org_centers = prev_centers[center]
                cur_centers = self.centers[center]
                if np.sum((cur_centers - org_centers) / org_centers * 100.0) > self.tolerance:
                    optimized = False
            if optimized:
                break

    def predict(self, p_data):
        for center in self.centers:
            distances = [np.linalg.norm(p_data - self.centers[center])]
        index = distances.index(min(distances))
        return index


if __name__ == '__main__':
    x = np.array([[1, 2], [1.5, 1.8], [5, 8], [8, 8], [1, 0.6], [9, 11]])
    k_means = Kmeans(k=2)
    k_means.fit(x)
    for i in range(len(k_means.centers)):  # 绘制中心点
        pyplot.scatter(k_means.centers[i][0], k_means.centers[i][1], marker='*', s=150)

    for i in range(len(k_means.clf)):
        for point in k_means.clf[i]:  # 绘制所有数据点
            pyplot.scatter(point[0], point[1], c=('r' if i == 0 else 'b'))

    predict = [[2, 1], [6, 9]]  # 给出新的数据点，进行预测并绘制
    for feature in predict:
        cat = k_means.predict(predict)
        pyplot.scatter(feature[0], feature[1], c=('r' if cat == 0 else 'b'), marker='x')

    pyplot.show()