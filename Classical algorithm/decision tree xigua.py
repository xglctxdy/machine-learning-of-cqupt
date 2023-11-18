import numpy as np
import math
import pandas as pd

dataSet = pd.read_csv('../data/xigua_data3.0.csv')
dataSet = dataSet.drop(["密度", "含糖率", "编号"], axis=1)
labels = dataSet.columns.tolist()
labels = labels[0:len(labels) - 1]
dataSet = np.array(dataSet)
split = int(len(dataSet) * 0.8)
test_set = dataSet[split:]
dataSet = dataSet[:split]

print(dataSet)

class DMTNode:
    def __init__(self, label, leaf=False, result="NaN"):
        # cur_label表示当前这个节点是以labels[cur_label]属性来进行划分
        # 如果为叶子节点规定label为0
        self.cur_label = label
        self.node = {}
        self.is_leaf = leaf
        self.result = result

    def create_child(self, factor, childnode):
        self.node[factor] = childnode

    def show(self):
        if self.is_leaf:
            print(self.result)
            return
        print(labels[self.cur_label], end="")
        for key in self.node.keys():
            print(" ", key, end="")
            self.node[key].show()

    def find_next(self, factor):
        if self.is_leaf:
            # print("判断结果为", self.result)
            return self.result
        return self.node[factor[self.cur_label]].find_next(factor)


def compute_entd(data_ava):
    # 这个函数是计算entd,data_ava表示目前在dataset中能访问的数据，其是一个bool向量
    temp = {}
    for i in range(len(data_ava)):
        if data_ava[i]:
            if dataSet[i][-1] not in temp:
                temp[dataSet[i][-1]] = 1
            else:
                temp[dataSet[i][-1]] += 1
    total = 0
    for value in temp.values():
        total += value
    entd = 0
    for value in temp.values():
        entd += -(value / total) * math.log(value / total, 2)
    return entd


def compute_info_Gain(data_ava, cur_label):
    # data_ava是一个bool向量，表示dataset中哪些数据是可访问的，cur_label表示当前的属性
    # 计算当前属性cur_label的信息增益
    # print("当前属性为", labels[cur_label])
    # temp是一个字典，key表示当前属性的具体选项，value表示表示具体选项的数目
    # 比如属性是颜色，key表示红，黄，其对应的value表示红色的数目
    temp = {}
    # data_total表示当前要划分的集合有多少个数据
    data_total = 0
    # ans表示当前属性的固有值
    ans = 0
    for j in range(len(data_ava)):
        if data_ava[j]:
            data_total += 1
            if dataSet[j][cur_label] not in temp:
                temp[dataSet[j][cur_label]] = 1
            else:
                temp[dataSet[j][cur_label]] += 1
    for key in temp.keys():
        data_label_ava = np.full(len(data_ava), False, dtype=bool)
        for i in range(len(data_ava)):
            if data_ava[i]:
                if dataSet[i][cur_label] == key:
                    data_label_ava[i] = True
        ans += (temp[key] / data_total) * compute_entd(data_label_ava)
        # print("当前", key, "信息增益为")
        # print(compute_entd(data_label_ava))
    return ans


def create_node(data_ava, label_ava):
    # 没有可用特征
    if (label_ava == np.full(len(label_ava), False, dtype=bool)).all():
        # ans的key表示判断结果，value表示该结果的数量
        ans = {}
        for i in range(len(data_ava)):
            if data_ava[i]:
                if dataSet[i][-1] not in ans:
                    ans[dataSet[i][-1]] = 1
                else:
                    ans[dataSet[i][-1]] += 1
        result = max(ans, key=lambda x: ans[x])
        return DMTNode(0, True, result)
    # 计算Ent(D)
    entd = compute_entd(data_ava)
    # 当前划分内数据的最终的结果完全一样
    if entd == 0.0:
        result = "NaN"
        for i in range(len(data_ava)):
            if data_ava[i]:
                result = dataSet[i][-1]
                break
        return DMTNode(0, True, result)

    # 计算每个属性的信息增益
    # temp的key表示该属性的下标，value表示其信息增益
    temp = {}
    for i in range(len(label_ava)):
        if label_ava[i]:
            temp[i] = entd - compute_info_Gain(data_ava, i)

    for key in temp.keys():
        print(labels[key], "的信息增益为", temp[key])

    cur_factor = max(temp, key=lambda x: temp[x])
    print("我们选择", labels[cur_factor], "作为当前划分点")
    nex_label = label_ava.copy()
    nex_label[cur_factor] = False
    # 从这里开始,temp的key表示当前选中属性中的选项，比如颜色中的红色，黄色
    temp = set()
    for i in range(len(data_ava)):
        if data_ava[i]:
            temp.add(dataSet[i][cur_factor])
    # root是该create_node函数所创造出来的节点
    root = DMTNode(cur_factor)
    # 下面这个for loop是为了对data_ava进行划分
    for i in temp:
        nex_data = data_ava.copy()
        for j in range(len(data_ava)):
            if data_ava[j] and (dataSet[j][cur_factor] != i):
                nex_data[j] = False
        root.create_child(i, create_node(nex_data, nex_label))
    return root


data_train = np.full(len(dataSet), True, dtype=bool)
label_train = np.full(len(labels), True, dtype=bool)
root = create_node(data_train, label_train)
times = 0
cor_times = 0
for i in test_set:
    times += 1
    # print(i)
    test = i[0:len(i) - 1]
    # print(test)
    result = root.find_next(test)
    print("预测结果为", result, "正确结果为", i[-1])
    if result == i[-1]:
        cor_times += 1
print("正确率为", cor_times / times)
