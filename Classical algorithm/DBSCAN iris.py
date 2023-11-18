import math
import numpy as np
import pandas as pd
import random

path = "../data/Iris_data/iris.csv"
dataset = pd.read_csv(path)
dataset = dataset.drop(['Unnamed: 0', 'Species'], axis=1)
labels = dataset.columns.values
dataset = dataset.values
"""print(dataset)
print(labels)"""
# cu表示最后我们分好类的结果，例如cu[0]表示以ans[0]为中心点的所有元素的一个列表
cu = []
# MinPts表示当范围内有MinPts个点时该点才被认定为中心点（包括自己），r表示范围（划定圆的半径）
MinPts = 4
r = 0.8


def distance(cur, target):
    # 计算两个点之间的欧拉距离
    # cur和target都是一个一维向量，代表当前的坐标
    dis = 0
    for i in range(len(cur)):
        dis += (cur[i] - target[i]) ** 2
    dis = math.sqrt(dis)
    return dis


class mode:
    def __init__(self):
        # sign用来给数据打上标签
        # 0代表中心点，1代表从属点，2代表噪声
        self.sign = []
        for i in range(3):
            self.sign.append(set())

    def marking_single(self, cur_point, only_Master=False):
        # 用来给当前点打上标签，cur_point代表当前这个点在dataset中的索引
        # only_Master为真，表示当前只给master点（中心点）打标签
        # total_pts代表目前有多少个点在其表示范围内
        total_pts = 1
        if only_Master:
            for i in range(len(dataset)):
                if i != cur_point:
                    if distance(dataset[i], dataset[cur_point]) <= r:
                        total_pts += 1
            if total_pts >= MinPts:
                return 0
        else:
            # is_master表示范围内是否有中心点存在
            is_master = False
            for i in range(len(dataset)):
                if i != cur_point:
                    if distance(dataset[i], dataset[cur_point]) <= r:
                        total_pts += 1
                        if i in self.sign[0]:
                            is_master = True
            if is_master:
                return 1
            else:
                return 2
        return -1

    def marking(self):
        # 用来给每一个点打上标签
        # visited用来记录该点是否被访问过
        visited = set()

        for i in range(len(dataset)):
            if i in visited:
                continue
            # 第一遍只标记中心点
            label = self.marking_single(i, True)
            if label != -1:
                self.sign[label].add(i)
                visited.add(i)

        for i in range(len(dataset)):
            if i in visited:
                continue
            label = self.marking_single(i)
            if label != -1:
                self.sign[label].add(i)
                visited.add(i)

        if len(visited) != len(dataset):
            print("仍存在未被访问过的点！请检查程序正确性！")
            exit()

    def relevance(self, cur_point):
        # 这个函数用来计算有多少个点属于cur_point的范围内
        point = []
        for i in range(len(dataset)):
            if i != cur_point:
                if distance(dataset[i], dataset[cur_point]) <= r:
                    point.append(i)
        return point

    def merge_point(self, cur_point, visited, cur_cu):
        nex = self.relevance(cur_point)
        for i in nex:
            if i not in visited:
                visited.add(i)
                cur_cu.append(i)
                self.merge_point(i, visited, cur_cu)

    def merge(self):
        # 这个函数用来合并簇
        visited = set()
        # 首先对中心点进行分类
        for i in self.sign[0]:
            if i not in visited:
                # 新建一个簇
                cur_cu = [i]
                visited.add(i)
                self.merge_point(i, visited, cur_cu)
                cu.append(cur_cu)


class SC:
    def buxiangsidu(self, cur_data, cur_cu, sign=False):
        # 计算一个样本和一个指定的簇之间的不相似度
        # sign表示是否计算簇内不相似度
        total = 0
        for i in cu[cur_cu]:
            total += distance(dataset[cur_data], dataset[i])
        if sign:
            total /= (len(cu[cur_cu]) - 1)
        else:
            total /= len(cu[cur_cu])
        return total

    def assess(self, cu):
        # 求轮廓系数
        total_si = 0
        for i in range(len(dataset)):
            # 计算样本i的轮廓系数
            # 首先计算ai，cur_cu表示该点当前所在簇
            cur_cu = 0
            for j in range(len(cu)):
                if i in cu[j]:
                    cur_cu = j
                    break
            ai = self.buxiangsidu(i, cur_cu, True)
            # 下面计算bi
            total_bi = []
            for j in range(len(cu)):
                # 代表j不属于当前簇
                if j != cur_cu:
                    total_bi.append(self.buxiangsidu(i, j))
            bi = min(total_bi)
            total_si += (bi - ai) / max(ai, bi)
        total_si /= len(dataset)
        print("平均轮廓系数为", total_si)


DB = mode()
DB.marking()
DB.merge()
# print(cu)
print("当前有", len(cu), "个簇")
sc = SC()
sc.assess(cu)
