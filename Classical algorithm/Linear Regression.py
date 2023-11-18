import matplotlib
import random
import torch
from d2l import torch as d2l
import matplotlib.pyplot as plt


def generate_data(w, b, num):
    x = torch.normal(0, 1, (num, len(w)))
    y = torch.matmul(x, w) + b
    y += torch.normal(0, 0.1, y.shape)
    return x, y.reshape(-1, 1)


def data_reader(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    random.shuffle(indices)
    for i in range(0, num_examples, batch_size):
        if i + batch_size < num_examples:
            random.shuffle(indices)
            i = 0
        batch_indices = torch.tensor(indices[i: i + batch_size])
        yield features[batch_indices], labels[batch_indices]


def linreg(X, w, b):
    return torch.matmul(X, w) + b


def squared_loss(y_hat, y):
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2


def sgd(params, lr, batch_size):
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad / batch_size
            param.grad.zero_()


true_w = torch.tensor([2, -1.0])
true_b = 4.2
features, labels = generate_data(true_w, true_b, 1000)
print('features:', features[0], '\nlabel:', labels[0])
w = torch.normal(0, 0.01, size=(2, 1), requires_grad=True)
b = torch.zeros(1, requires_grad=True)
d2l.set_figsize()
plt.scatter(features[:, 1].detach().numpy(), labels.detach().numpy(), 1)
plt.show()
lr = 0.03
batch_size = 10
num_epochs = 3
net = linreg
loss = squared_loss

for epoch in range(num_epochs):
    for X, y in data_reader(batch_size, features, labels):
        l = loss(net(X, w, b), y)  # X和y的小批量损失
        # 因为l形状是(batch_size,1)，而不是一个标量。l中的所有元素被加到一起，
        # 并以此计算关于[w,b]的梯度
        l.sum().backward()
        sgd([w, b], lr, batch_size)  # 使用参数的梯度更新参数
    with torch.no_grad():
        train_l = loss(net(features, w, b), labels)
        print(f'epoch {epoch + 1}, loss {float(train_l.mean()):f}')

print(f'w的估计误差: {true_w - w.reshape(true_w.shape)}')
print(f'b的估计误差: {true_b - b}')
