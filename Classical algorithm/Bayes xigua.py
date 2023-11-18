import numpy as np
import pandas as pd

# 数据预处理
dataset = pd.read_csv('../data/xigua_data3.0.csv')
dataset = dataset.drop(['编号', '密度', '含糖率'], axis=1)
label = dataset.columns.values
label = label[0:len(label) - 1]
dataset = dataset.values
"""print(label)
print(dataset)"""


# y的类别，比如是否为好瓜
class ResultY:
    def __init__(self):
        # classes_y来储存y一共有多少类，每个类有多少个元素
        self.classes_y = {}
        # prob_y用来储存y的结果为yi时的概率为多少
        self.prob_y = {}

    def initialize(self, data):
        for i in data[:, -1]:
            if i not in self.classes_y:
                self.classes_y[i] = 1
            else:
                self.classes_y[i] += 1
        total = 0
        for val in self.classes_y.values():
            total += val
        # 将total加上y的所有种类数，就是拉普拉斯平滑
        total += len(self.classes_y)
        for key in self.classes_y.keys():
            # 引入拉普拉斯平滑
            self.prob_y[key] = (self.classes_y[key] + 1) / total

    def show(self):
        # 当前y的种类和数量，概率分别为
        print("种类  数量  概率")
        for key in self.classes_y.keys():
            print(key, "  ", self.classes_y[key], "  ", self.prob_y[key])


class Bayes:
    def __init__(self, t, label_num):
        self.result_y = {}
        # result第一个索引代表是好瓜还是坏瓜，第二个索引代表瓜的属性，如颜色，第三个索引才代表类别，如颜色为青绿
        for i in t.keys():
            self.result_y[i] = np.array([{}] * label_num)

    def create(self):
        for key in self.result_y.keys():
            # cur_dataset是一个bool向量，代表数据里面哪些值的最后结果是’好瓜‘/’坏瓜‘
            cur_dataset = dataset[:, -1] == key
            # 遍历每个特征，比如遍历颜色
            for i in range(len(label)):
                self.result_y[key][i] = self.count_prob(i, cur_dataset)
                # print(self.result_y[key][i])

    def count_prob(self, cur_label, cur_dataset):
        # cur_total是指当前好瓜或者坏瓜一共有多少个
        cur_total = np.count_nonzero(cur_dataset)
        temp = {}
        for i in range(len(cur_dataset)):
            if cur_dataset[i]:
                if dataset[i][cur_label] not in temp:
                    temp[dataset[i][cur_label]] = 1
                else:
                    temp[dataset[i][cur_label]] += 1
        # 引入拉普拉斯平滑
        cur_total += len(temp)
        for key in temp.keys():
            temp[key] += 1
            temp[key] /= cur_total
        # 设立标志变量，表示该点不存在
        temp['Not_Found_zyh'] = 1.0 / cur_total
        return temp

    def show(self):
        for key in self.result_y.keys():
            print("当前目标属性为", key)
            for i in range(len(self.result_y[key])):
                print("当前类别为", label[i])
                for x in self.result_y[key][i].keys():
                    print("当前属性为", x, " 当前属性的概率为", self.result_y[key][i][x])

    def find(self, cur_data, y):
        # 这个temp用来存放每个目标的概率可能性
        temp = {}
        for key in self.result_y.keys():
            # cur prob是当前概率
            cur_prob = y.prob_y[key]
            for i in range(len(cur_data)):
                if cur_data[i] not in self.result_y[key][i]:
                    cur_prob *= self.result_y[key][i]['Not_Found_zyh']
                else:
                    cur_prob *= self.result_y[key][i][cur_data[i]]
            temp[key] = cur_prob
        result = max(temp, key=lambda x: temp[x])
        return result


y = ResultY()
y.initialize(dataset)
bayes = Bayes(y.classes_y, len(label))
bayes.create()
# 查看生成的数据
# bayes.show()
total_times = len(dataset)
right_times = 0
for i in range(len(dataset)):
    cur_data = dataset[i][0:len(dataset[i]) - 1]
    result = bayes.find(cur_data, y)
    print("预测值为", result, " 真实值为", dataset[i][-1])
    if result == dataset[i][-1]:
        right_times += 1
print("正确率为", right_times / total_times)
