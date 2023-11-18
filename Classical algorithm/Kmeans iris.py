import math
import numpy as np
import pandas as pd
import random
import matplotlib as plt

path = "../data/Iris_data/iris.csv"
dataset = pd.read_csv(path)
dataset = dataset.drop(['Unnamed: 0', 'Species'], axis=1)
labels = dataset.columns.values
dataset = dataset.values
"""print(dataset)
print(labels)"""
# ans表示最后的目标中心点
ans = []
# cu表示最后我们分好类的结果，例如cu[0]表示以ans[0]为中心点的所有元素的一个列表
cu = []
# k表示一共分多少簇，threshold表示阈值为多少，times_limit表示最多迭代多少次
k = 3
threshold = 0.0001
times_limit = 20


def initialize(ans, k, num_data, cu):
    ans.clear()
    cu.clear()
    temp = random.sample(range(0, num_data), k)
    for i in temp:
        ans.append(dataset[i])
        cu.append([])


def distance(cur, target):
    # cur和target都是一个一维向量，代表当前的坐标
    # 计算两个点之间的欧拉距离
    dis = 0
    for i in range(len(cur)):
        dis += (cur[i] - target[i]) ** 2
    dis = math.sqrt(dis)
    return dis


def skewing(cur_ans, ans):
    # cur_ans和ans都是二维向量，每一行代表一个坐标
    # 计算当前中心点和上一次中心点之间的偏移量
    dis = 0
    for i in range(len(cur_ans)):
        dis += distance(cur_ans[i], ans[i])
    return dis


def train():
    # cu用来存储当前这个簇下每个元素的下标号
    # 将我们的簇全部清空用来存放新的分类结果
    for i in range(len(cu)):
        cu[i].clear()
    for i in range(len(dataset)):
        temp = []
        for j in range(len(ans)):
            temp.append(distance(dataset[i], ans[j]))
        # 返回temp中最大的那个元素的下标i，就是距离i中心点最近，归到第i簇中
        cu[temp.index(min(temp))].append(i)
    # 新一轮的簇已经分类完成，现在我们重新计算中心点
    # cur_ans表示我们目前新计算出来的中心点
    cur_ans = ans.copy()
    for i in range(len(cu)):
        # 如果当前的簇没有任何点归到这一类，则直接将之前的中心点赋给cur_ans
        if len(cu[i]) == 0:
            continue
        else:
            cur_ans[i] = [0 for i in range(len(ans[i]))]
            for j in cu[i]:
                # 遍历这个簇中的每个元素
                for z in range(len(cur_ans[i])):
                    cur_ans[i][z] += dataset[j][z]
            # 取平均值
            for j in range(len(cur_ans[i])):
                cur_ans[i][j] /= len(cu[i])
    # print(cu)
    if skewing(cur_ans, ans) > threshold:
        # print(skewing(cur_ans, ans))
        # 表示仍然需要继续迭代，将新的中心点赋值给ans
        for i in range(len(ans)):
            for j in range(len(ans[0])):
                ans[i][j] = cur_ans[i][j]
        # print("仍需迭代")
        return True
    return False


def buxiangsidu(cur_data, cur_cu, sign=False):
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


def SC():
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
        ai = buxiangsidu(i, cur_cu, True)
        # 下面计算bi
        total_bi = []
        for j in range(len(cu)):
            # 代表j不属于当前簇
            if j != cur_cu:
                total_bi.append(buxiangsidu(i, j))
        bi = min(total_bi)
        total_si += (bi - ai) / max(ai, bi)
    total_si /= len(dataset)
    print("平均轮廓系数为", total_si)


# cur_dis表示当前中心点与上一次中心点的偏移值，cur_times表示目前已迭代多少次
sign = True
cur_times = 0
initialize(ans, k, len(dataset), cu)
while sign and cur_times <= times_limit:
    sign = train()
    cur_times += 1
print("已经迭代", cur_times, "次")
print("选取的中心点为", ans)
print("最终划分的簇为", cu)
SC()
