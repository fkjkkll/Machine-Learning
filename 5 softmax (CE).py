# -*- coding: utf-8 -*-
"""
softmax
hypothesis: h = softmax(np.matmul(x, theta))
loss function: cross entropy
Created on Sun Aug 29 12:37:32 2021
@author: lee
"""

import numpy as np
import matplotlib.pyplot as plt

#%% 数据生成
data_size = 360
m = 4
cov = [[1, 0],[0, 1]]
x = np.random.multivariate_normal([0, 0], cov, data_size//9)
x = np.row_stack((x, np.random.multivariate_normal([-m, m], cov, data_size//9)))
x = np.row_stack((x, np.random.multivariate_normal([m, -m], cov, data_size//9)))
x = np.row_stack((x, np.random.multivariate_normal([m, 0], cov, data_size//9)))
x = np.row_stack((x, np.random.multivariate_normal([m, m], cov, data_size//9)))
x = np.row_stack((x, np.random.multivariate_normal([0, m], cov, data_size//9)))
x = np.row_stack((x, np.random.multivariate_normal([-m, 0], cov, data_size//9)))
x = np.row_stack((x, np.random.multivariate_normal([-m, -m], cov, data_size//9)))
x = np.row_stack((x, np.random.multivariate_normal([0, -m], cov, data_size//9)))
x = np.column_stack((np.ones(data_size), x))
y = np.zeros((data_size, 3))
y[:data_size//3, 0] = 1
y[data_size//3:2*data_size//3, 1] = 1
y[2*data_size//3:data_size, 2] = 1

#%% 参数设定
theta = np.random.randn(3, 3)

#%% 超参数设定
train_count = 1000
batch_size = 64
lr = 1e-3

#%% 假说函数
def softmax(x):
    offset = np.max(x, 1)
    x = x - offset.reshape(-1, 1)
    x = np.exp(x)
    return x/np.sum(x, 1).reshape(-1, 1)

def calHypo(x, theta):
    return softmax(np.matmul(x, theta))

#%% 损失函数
def calLoss(x, y, theta):
    h = calHypo(x, theta)
    # 最大化似然，加个负号最小化损失
    loss = - np.log(h[np.arange(data_size), np.argmax(y, 1)])
    return np.sum(loss)/y.shape[0]

#%% 准确率
def accuracy(x, y, theta):
    h = calHypo(x, theta)
    compare = (np.argmax(h, 1) == np.argmax(y, 1))
    return np.sum(compare)/x.shape[0]

#%% 显示
x1 = [[i / 10] * (160) for i in range(-80, 80)]
x1 = np.array(x1)
x1 = x1.flatten()
x2 = [i / 10 for i in range(-80, 80)] * (160)
x2 = np.array(x2)
x0 = np.ones(len(x1))
xt = np.column_stack((x0, x1, x2))
fig = plt.figure()
ax1 = fig.add_subplot(1, 2, 1)
ax2 = fig.add_subplot(1, 2, 2)
loss_list = []

#%% 训练
for i in range(1, train_count + 1):
    mask = np.random.choice(data_size, batch_size)
    train_x, train_y = x[mask], y[mask]
    h = calHypo(train_x, theta)
    # 通过这种操作批量化处理一批数据而不必一个一个去改变theta
    h[np.arange(batch_size), np.argmax(train_y, 1)] -= 1
    grad = np.matmul(train_x.T, h)
    theta = theta - lr*grad

    if i % (train_count // 20) == 0:
        loss = calLoss(x, y, theta)
        loss_list.append(loss)
        
        t_data = np.argmax(calHypo(xt, theta), 1)
        ax1.cla()
        mask0 = [i for i in range(len(t_data)) if t_data[i] == 0]
        mask1 = [i for i in range(len(t_data)) if t_data[i] == 1]
        mask2 = [i for i in range(len(t_data)) if t_data[i] == 2]
        ax1.scatter(xt[mask0, 1], xt[mask0, 2], color='blue', marker='s', s=10)
        ax1.scatter(xt[mask1, 1], xt[mask1, 2], color='red', marker='s', s=10)
        ax1.scatter(xt[mask2, 1], xt[mask2, 2], color='green', marker='s', s=10)
        data_size = x.shape[0]
        ax1.scatter(x[:data_size//3, 1], x[:data_size//3, 2], color='darkblue')
        ax1.scatter(x[data_size//3:2*data_size//3, 1], x[data_size//3:2*data_size//3, 2], color='darkred')
        ax1.scatter(x[2*data_size//3:data_size, 1], x[2*data_size//3:data_size, 2], color='darkgreen')
        ax1.set_title(i)
        
        ax2.cla()
        ax2.plot(range(len(loss_list)), loss_list)
        ax2.set_title('loss:' + str(loss))
        
        print('accuracy: ', accuracy(x, y, theta))
        plt.pause(0.1)
