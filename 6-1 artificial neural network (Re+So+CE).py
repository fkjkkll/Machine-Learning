# -*- coding: utf-8 -*-
"""
artificial neural network
hypothesis: relu -> softmax
loss function: cross entropy
Created on Mon Aug 30 14:06:45 2021
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

#%% 参数设定(设成全1，出问题：会退化为线性模型)
w = np.random.randn(3, 5)
v = np.random.randn(5, 3)

#%% 超参数设定
train_count = 1000
batch_size = 64
lr = 1e-2

#%% 假说函数
def relu(x):
    if not hasattr(relu, 'mask'):
        relu.mask = np.zeros_like(x)
    relu.mask = (x > 0)
    return x * relu.mask

def softmax(x):
    offset = np.max(x, 1)
    x = x - offset.reshape(-1, 1)
    x = np.exp(x)
    return x/np.sum(x, 1).reshape(-1, 1)

def calHypo(x, theta):
    _w, _v = theta[0], theta[1]
    z_hat = np.matmul(x, _w)
    z = relu(z_hat)
    h_hat = np.matmul(z, _v)
    h = softmax(h_hat)
    return h

#%% 损失函数
def calLoss(x, y, theta):
    h = calHypo(x, theta)
    loss = - np.log(h[np.arange(x.shape[0]), np.argmax(y, 1)])
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
    # 正向传播
    z_hat = np.matmul(train_x, w)
    z = relu(z_hat)
    h_hat = np.matmul(z, v)
    h = softmax(h_hat)
    # 反向传播更新参数
    grad_v = np.matmul(z.T, (h - train_y))
    grad_w = np.matmul(train_x.T, (np.matmul((h - train_y), v.T) * relu.mask))
    v = v - lr * grad_v
    w = w - lr * grad_w
    
    if i % (train_count // 20) == 0:
        loss = calLoss(x, y, [w, v])
        loss_list.append(loss)
        
        t_data = np.argmax(calHypo(xt, [w, v]), 1)
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
        
        print('accuracy: ', accuracy(x, y, [w, v]))
        plt.pause(0.1)











