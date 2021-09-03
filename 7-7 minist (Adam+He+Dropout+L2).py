# -*- coding: utf-8 -*-
"""
test on dataset
hypothesis: relu -> softmax
loss function: cross entropy
Created on Tue Aug 31 14:40:55 2021
@author: lee
"""
import numpy as np
import matplotlib.pyplot as plt
import idx2numpy

#%% 数据生成
def deal_x(x):
    return x.reshape(-1, 28*28) / 255

def deal_y(y):
    _y = np.zeros((y.shape[0], 10))
    _y[np.arange(y.shape[0]), y] = 1
    return _y

data_set = 'mnist' # mnist or fmnist

train_x = idx2numpy.convert_from_file('data/'+ data_set + '/train-images-idx3-ubyte')
train_y = idx2numpy.convert_from_file('data/'+ data_set + '/train-labels-idx1-ubyte')
test_x = idx2numpy.convert_from_file('data/'+ data_set + '/t10k-images-idx3-ubyte')
test_y = idx2numpy.convert_from_file('data/'+ data_set + '/t10k-labels-idx1-ubyte')

train_x = deal_x(train_x)
test_x = deal_x(test_x)
train_y = deal_y(train_y)
test_y = deal_y(test_y)
data_size = train_x.shape[0]

#%% 参数设定(He)
w = np.random.randn(784, 128) * np.sqrt(2/784)
v = np.random.randn(128, 10) * np.sqrt(2/128)

#%% 超参数设定
train_count = 10000
batch_size = 128
lr = 5e-3
keep_ratio = 0.5
gamma = 5e-4

# Adam
beta1 = 0.9
beta2 = 0.999
vdw = np.zeros_like(w)
sdw = np.zeros_like(w)
vdv = np.zeros_like(v)
sdv = np.zeros_like(v)

#%% 假说函数
def relu(x):
    if not hasattr(relu, 'fmask'):
        relu.mask = np.zeros_like(x)
    relu.mask = (x > 0)
    return x * relu.mask

def dropout(x):
    if not hasattr(dropout, 'mask'):
        dropout.mask = None
    dropout.mask = np.random.rand(*x.shape)
    dropout.mask = (dropout.mask < keep_ratio)
    return x * dropout.mask

def softmax(x):
    offset = np.max(x, 1)
    x = x - offset.reshape(-1, 1)
    x = np.exp(x)
    return x/np.sum(x, 1).reshape(-1, 1)

def calHypo(x, theta):
    _w, _v = theta[0], theta[1]
    z_hat = np.matmul(x, _w)
    z = relu(z_hat) * keep_ratio # dropout测试时加上
    h_hat = np.matmul(z, _v)
    h = softmax(h_hat)
    return h

#%% 损失函数
def calLoss(x, y, theta):
    h = calHypo(x, theta)
    loss = - np.log(h[np.arange(x.shape[0]), np.argmax(y, 1)])
    # loss += 0.5 * gamma * (np.sum(theta[0]**2) + np.sum(theta[1]**2)) # L2正则项
    return np.sum(loss)/y.shape[0]

#%% 准确率
def accuracy(x, y, theta):
    h = calHypo(x, theta)
    compare = (np.argmax(h, 1) == np.argmax(y, 1))
    return np.sum(compare)/x.shape[0]

#%% 显示
fig = plt.figure()
ax1 = fig.add_subplot(1, 2, 1)
ax2 = fig.add_subplot(1, 2, 2)
test_acc_list = []
train_acc_list = []
test_loss_list = []
train_loss_list = []

#%% 训练
for i in range(1, train_count + 1):
    mask = np.random.choice(data_size, batch_size)
    batch_train_x, batch_train_y = train_x[mask], train_y[mask]
    # 正向传播
    z_hat = np.matmul(batch_train_x, w)
    z = relu(z_hat)
    z = dropout(z)
    h_hat = np.matmul(z, v)
    h = softmax(h_hat)
    
    # 反向传播更新参数
    grad_v = np.matmul(z.T, (h - batch_train_y)) + gamma*v
    grad_w = np.matmul(batch_train_x.T, (np.matmul((h - batch_train_y), v.T) * relu.mask * dropout.mask)) + gamma*w
    
    vdv = beta1*vdv + (1-beta1)*grad_v
    sdv = beta2*sdv + (1-beta2)*grad_v**2
    vcdv = vdv / (1 - beta1**(i+1))
    scdv = sdv / (1 - beta2**(i+1))
    v = v - lr * (vcdv/(np.sqrt(scdv) + 1e-8))
    
    vdw = beta1*vdw + (1-beta1)*grad_w
    sdw = beta2*sdw + (1-beta2)*grad_w**2
    vcdw = vdw / (1 - beta1**(i+1))
    scdw = sdw / (1 - beta2**(i+1))
    w = w - lr * (vcdw/(np.sqrt(scdw) + 1e-8))
    
    if i % (train_count // 20) == 0:
        lr = lr * 0.9
        
        train_acc = accuracy(train_x, train_y, [w, v])
        train_acc_list.append(train_acc)
        test_acc = accuracy(test_x, test_y, [w, v])
        test_acc_list.append(test_acc)
        
        ax1.cla()
        ax1.plot(range(len(train_acc_list)), train_acc_list, label='train')
        ax1.plot(range(len(test_acc_list)), test_acc_list, label='test')
        ax1.set_ylabel('accuracy')
        ax1.legend()
        ax1.set_title('test acc: ' + str(test_acc) + '\ntrain acc' + str(train_acc))
        
        train_loss = calLoss(train_x, train_y, [w, v])
        train_loss_list.append(train_loss)
        test_loss = calLoss(test_x, test_y, [w, v])
        test_loss_list.append(test_loss)
        
        ax2.cla()
        ax2.plot(range(len(train_loss_list)), train_loss_list, label='train')
        ax2.plot(range(len(test_loss_list)), test_loss_list, label='test')
        ax2.set_ylabel('loss')
        ax2.legend()
        ax2.set_title('train_count: ' + str(i) + '/' + str(train_count))
        
        plt.pause(0.1)

#%% 单例测试
dic = {0:'T-shirt', 1:'Trouser', 2:'Pullover', 3:'Dress', 4:'Coat', 5:'Sandal',
       6:'Shirt', 7:'Sneaker', 8:'Bag', 9:'Ankle boot'}

def testPicture():
    index = np.random.randint(0, 10000)
    data = test_x[index].reshape(1, -1)
    h = calHypo(data, [w, v])
    h = np.argmax(h)
    img = data.reshape((28, 28))
    plt.imshow(img)
    if data_set == 'mnist':
        plt.title(h)
    else:
        plt.title(dic[h])


















