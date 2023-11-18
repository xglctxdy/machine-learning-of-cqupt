import numpy as np
import pandas as pd
import re

qiefen = 5000
path = '../data/SMSSpamCollection'

total_set = pd.read_csv(path, sep='\t', header=None)
total_set = total_set.replace({'ham': 0, 'spam': 1})
total_set = total_set.values
np.random.shuffle(total_set)
train_data = total_set[0:qiefen]
test_data = total_set[qiefen:-1]


class ResultY:
    def __init__(self):
        # classes_y来储存y一共有多少类，每个类有多少个元素
        self.classes_y = {}
        # prob_y用来储存y的结果为yi时的概率为多少
        self.prob_y = {}

    def initialize(self, data):
        for i in data[:, 0]:
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
    def __init__(self, t):
        self.result_y = {}
        # result第一个索引代表是有害信息还是无害，第二个索引代表单词，其对应的为该单词出现字数
        for i in t.keys():
            self.result_y[i] = {}

    def split(self, key, i, message):
        # print(message)
        # temp用来存储message中被分割好的单词
        temp = re.split('\W+', message)
        for i in temp:
            if i not in self.result_y[key]:
                self.result_y[key][i] = 1
            else:
                self.result_y[key][i] += 1
            # print(i)
        return

    def train(self, data_set):
        for key in self.result_y.keys():
            # cur_suoyin是当前的一个索引，代表哪些数据的标签就是key
            cur_suoyin = data_set[:, 0] == key
            # print(cur_suoyin)
            # print(np.count_nonzero(cur_suoyin))
            for i in range(len(data_set)):
                if cur_suoyin[i]:
                    self.split(key, i, data_set[i][1])

    def test(self, data_set, y):
        # num_total是测试样例总数，cur_total是正确的样例数
        num_total = len(data_set)
        cur_total = 0
        for i in data_set:
            result = self.test_self(i[1])
            if result == i[0]:
                cur_total += 1
        print("正确率为", cur_total / num_total)

    def test_self(self, message, show=False):
        # temp代表该信息属于哪一类的概率
        temp = {}
        for key in self.result_y.keys():
            temp[key] = y.prob_y[key]
            words = re.split('\W+', message)
            for word in words:
                if word not in self.result_y[key]:
                    # 没找到就加入拉普拉斯平滑处理
                    temp[key] *= 1 / (y.classes_y[key] + len(self.result_y[key]))
                else:
                    temp[key] *= (1 + self.result_y[key][word]) / (y.classes_y[key] + len(self.result_y[key]))
        result = max(temp, key=lambda x: temp[x])
        if show:
            if result:
                print("该短信为垃圾短信")
            else:
                print("该短信为正常短信")
        return result


y = ResultY()
y.initialize(train_data)
y.show()
bayes = Bayes(y.classes_y)
bayes.train(train_data)
bayes.test(test_data, y)

message = "sadhiu idsahiuash hdsauihdas"
bayes.test_self(message, True)
