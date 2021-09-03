"""
Created on Sun Aug 29 09:51:22 2021

@author: lee
"""
import numpy as np
from utils.tool import Leeplot

#%% 数据生成
data_size = 360
m = 4
cov = [[1,0],[0,1]]
x = np.random.multivariate_normal([0,0], cov, data_size//9)
x = np.row_stack((x, np.random.multivariate_normal([-m,m], cov, data_size//9)))
x = np.row_stack((x, np.random.multivariate_normal([m,-m], cov, data_size//9)))
x = np.row_stack((x, np.random.multivariate_normal([m,0], cov, data_size//9)))
x = np.row_stack((x, np.random.multivariate_normal([m,m], cov, data_size//9)))
x = np.row_stack((x, np.random.multivariate_normal([0,m], cov, data_size//9)))
x = np.row_stack((x, np.random.multivariate_normal([-m,0], cov, data_size//9)))
x = np.row_stack((x, np.random.multivariate_normal([-m,-m], cov, data_size//9)))
x = np.row_stack((x, np.random.multivariate_normal([0,-m], cov, data_size//9)))
x = np.column_stack((np.ones(data_size), x))
y = np.zeros((data_size, 3))
y[:data_size//3, 0] = 1
y[data_size//3:2*data_size//3, 1] = 1
y[2*data_size//3:data_size, 2] = 1

#%% 参数设定
theta = np.ones((3,3))

#%% 超参数设定
train_count = 1000
batch_size = 64
lr = 1e-3

#%% 假说函数
def calHypo(x, theta):
    return np.matmul(x, theta.T)

#%% 损失函数
def calLoss(x, y):
    _h = np.matmul(x, theta.T)
    _loss = np.max(_h, 1) - _h[np.arange(data_size), np.argmax(y, 1)]
    return np.sum(_loss)/data_size

#%% 图像显示
lplt = Leeplot(x, y, -8, 8, -8, 8, calHypo, calLoss)

#%% 训练
for i in range(1, train_count+1):
    mask = np.random.choice(data_size, batch_size)
    train_x = x[mask]
    train_y = y[mask]
    h = np.matmul(train_x, theta.T)
    # 通过这种操作批量化处理一批数据而不必一个一个去改变theta
    mp = np.zeros_like(train_x)
    mp[np.arange(batch_size), np.argmax(h, 1)] += 1
    mp[np.arange(batch_size), np.argmax(train_y, 1).astype(int)] -= 1
    grad = np.matmul(mp.T, train_x)
    theta = theta - lr*grad

    if i % (train_count//20) == 0:
        lplt.show(i, theta)
        
    
    
