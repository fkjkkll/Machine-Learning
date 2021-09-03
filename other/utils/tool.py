# -*- coding: utf-8 -*-
"""
Created on Sun Aug 29 09:51:22 2021

@author: lee
"""
import numpy as np
import matplotlib.pyplot as plt

class Leeplot:
    '''
    x: data;
    y: label;
    xl: x range start;
    xr: x range end;
    yl: y range start;
    yr: y range end;
    calHypo: calculate hypothesis;
    calLoss: calculate loss;
    '''
    def __init__(self, x, y, xl, xr, yl, yr, calHypo, calLoss):
        x1 = [[i / 10] * ((-xl+xr)*10) for i in range(xl*10, xr*10)]
        x1 = np.array(x1)
        x1 = x1.flatten()
        x2 = [i / 10 for i in range(yl*10, yr*10)] * ((-yl+yr)*10)
        x2 = np.array(x2)
        x0 = np.ones(len(x1))
        
        self.x = x
        self.y = y
        self.calLoss = calLoss
        self.calHypo = calHypo
        self.xt = np.column_stack((x0, x1, x2))
        fig = plt.figure()
        self.ax1 = fig.add_subplot(1, 2, 1)
        self.ax2 = fig.add_subplot(1, 2, 2)
        self.loss_list = []
        
    def show(self, i, theta, interval=0.1):
        '''
        i: iterator count;
        theta: learnable parameters;
        interval: show interval;
        '''
        loss = self.calLoss(self.x, self.y)
        self.loss_list.append(loss)
        
        t_data = np.argmax(self.calHypo(self.xt, theta), 1)
        self.ax1.cla()
        mask0 = [i for i in range(len(t_data)) if t_data[i] == 0]
        mask1 = [i for i in range(len(t_data)) if t_data[i] == 1]
        mask2 = [i for i in range(len(t_data)) if t_data[i] == 2]
        self.ax1.scatter(self.xt[mask0, 1], self.xt[mask0, 2], color='blue', marker='s', s=10)
        self.ax1.scatter(self.xt[mask1, 1], self.xt[mask1, 2], color='red', marker='s', s=10)
        self.ax1.scatter(self.xt[mask2, 1], self.xt[mask2, 2], color='green', marker='s', s=10)
        data_size = self.x.shape[0]
        self.ax1.scatter(self.x[:data_size//3, 1], self.x[:data_size//3, 2], color='darkblue')
        self.ax1.scatter(self.x[data_size//3:2*data_size//3, 1], self.x[data_size//3:2*data_size//3, 2], color='darkred')
        self.ax1.scatter(self.x[2*data_size//3:data_size, 1], self.x[2*data_size//3:data_size, 2], color='darkgreen')
        self.ax1.set_title(i)
        
        self.ax2.cla()
        self.ax2.plot(range(len(self.loss_list)), self.loss_list)
        self.ax2.set_title('loss:' + str(loss))
        
        plt.pause(interval)
        
        
        
        
        
        
        
        
        
        