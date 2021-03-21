# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.

忽略飞行器姿态以及碰撞，忽略leader-follower，初步编队仿真尝试
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment
from matplotlib.animation import FuncAnimation
#import time
#import sys
#import mpl_toolkits.mplot3d.axes3d as p3
#import copy

NUM = 10 #设定无人机数目
MOVE_DISTANCE = 0.1

'''
邻接矩阵，规范同论文一致,leader为首行首列
A = 
[0,0,0,0,0,0,0,0,0,0]
[1,0,0,0,0,0,0,0,0,0]
[1,1,0,0,0,0,0,0,0,0]
[0,1,1,0,0,0,0,0,0,0]
[0,0,1,1,0,0,0,0,0,0]
[0,0,0,1,1,0,0,0,0,0]
[0,0,0,0,1,1,0,0,0,0]
[0,0,0,0,0,1,1,0,0,0]
[0,0,0,0,0,0,1,1,0,0]
[0,0,0,0,0,0,0,1,1,0]

度矩阵,仅考虑入度，leader为首行首列
B = diag(0,1,2,2,2,2,2,2,2,2)

L = B - A
-L特征值为：
diag(2,2,2,2,2,2,2,2,1,0)
'''



'''
预设几个队列坐标
'''
#随机生成无人机初始坐标
x_initial = np.random.randint(-6,6,10)
y_initial = np.random.randint(-6,6,10)
z_initial = np.random.randint(-6,6,10)

#十二面体坐标
dodecahedron = [[0,0,0],[1,1,-2],[1,-1,-2],[-1,1,-2],[-1,-1,-2],[1,1,-4],[1,-1,-4],[-1,1,-4],[-1,-1,-4],[0,0,-6]]
#T形
T_shape = [[0,0,0],[0,1,0],[0,2,0],[0,-1,0],[0,-2,0],[1,0,0],[2,0,0],[-1,0,0],[-2,0,0],[-3,0,0]]
#三角形
triangle = [[0,0,0],[2,0,0],[-2,0,0],[1,0,2],[-1,0,2],[0,0,4],[1,0,-2],[3,0,-2],[-1,0,-2],[-3,0,-2]]

#选择预设坐标
shape = triangle

#储存各无人机当前坐标
pot_list = []
for i in range(NUM):
    pot_list.append([x_initial[i],y_initial[i],z_initial[i]])
    
    
    
'''
定义一个无人机的类
'''
class follower():
    MOVE_DISTANCE = MOVE_DISTANCE
    aim_pot = None
    
    def __init__(self,id):
        self.id = id
        self.location = [x_initial[id],y_initial[id],z_initial[id]]

    def distance_calculate(self,shape):
        '''
        计算该无人机与各目标点间的距离,返回距离的一维数组
        '''
        location = self.location
        distance = []
        for each in shape:
            Distance = ((location[0] - each[0])**2 + (location[1] - each[1])**2 + (location[2] - each[2])**2)**0.5
            distance.append(Distance)
        return distance

    def aim_distance(self,aim):
        '''
        计算该无人机与目标坐标之间的距离，返回一个float距离
        '''
        location = self.location
        distance = ((location[0] - aim[0])**2 + (location[1] - aim[1])**2 + (location[2] - aim[2])**2)**0.5
        return distance

    def aim_location(self,shape,col_ind):
        '''
        通过匈牙利算法结果得到目标坐标,返回目标坐标（三维）
        '''
        aim_location = shape[col_ind[self.id]]
        return aim_location

    def angle_calculate(self,aim):
        '''
        计算起始坐标与目标坐标形成直线在x,y,z轴上夹角的cos值，返回一个三维列表
        '''
        location = self.location
        distance = self.aim_distance(aim)
        if distance >= MOVE_DISTANCE:
            cos_theta = [(aim[0] - location[0]) / distance,(aim[1] - location[1]) / distance,(aim[2] - location[2]) / distance] #两点所形成直线分别与x,y,z坐标形成夹角的cos值
        else:
            cos_theta = [0,0,0]
        return cos_theta
    
    def move(self,aim):
        cos_theta = self.angle_calculate(aim)
        distance = self.aim_distance(aim)
        if  distance >= self.MOVE_DISTANCE:
            self.location[0] = self.location[0] + MOVE_DISTANCE * cos_theta[0]
            self.location[1] = self.location[1] + MOVE_DISTANCE * cos_theta[1]
            self.location[2] = self.location[2] + MOVE_DISTANCE * cos_theta[2]
            pot_list[self.id] = self.location
        else:
            self.location[0] = aim[0]
            self.location[1] = aim[1]
            self.location[2] = aim[2]
            pot_list[self.id] = self.location

#生成无人机与各目标点距离的NUM*NUM的二维数组cost
obj_list = [follower(i) for i in range(NUM)]
cost = np.zeros([NUM,NUM])
for i in range(NUM):
    for j in range(NUM):
        trans = obj_list[i].distance_calculate(shape)
        cost[i][j] = trans[j]

#匈牙利算法分配路径,确定目标坐标
row_ind,col_ind = linear_sum_assignment(cost)
aim_list = []
for i in range(NUM):
    obj_list[i].aim_pot = obj_list[i].aim_location(shape,col_ind)
    aim_list.append(obj_list[i].aim_pot)
    


'''
绘图（有待改进）
'''
#确定最大值，来容纳移动时产生的坐标
N_list = []
for i in range(NUM):
    N_list.append((obj_list[i].aim_distance(aim_list[i]) / MOVE_DISTANCE))
N_MAX = int(max(N_list)) + 1
x_list = np.ndarray([N_MAX, NUM])
y_list = np.ndarray([N_MAX, NUM])
z_list = np.ndarray([N_MAX, NUM])

print(N_MAX)

for i in range(N_MAX):
    for j in range(NUM):
        obj_list[j].move(aim_list[j])
        x_list[i][j] = obj_list[j].location[0]
        y_list[i][j] = obj_list[j].location[1]
        z_list[i][j] = obj_list[j].location[2]

fig  = plt.figure(figsize=(10,10))
ax = fig.add_subplot(1,1,1,projection='3d')
point, = ax.plot(x_list[0,:],y_list[0,:],z_list[0,:],'r.', marker = '1')
ax.set_xlim3d([-10,10])
ax.set_xlabel('x')
ax.set_ylim3d([-10,10])
ax.set_ylabel('y')
ax.set_zlim3d([-10,10])
ax.set_zlabel('z')

def animate(i):
    point.set_xdata(x_list[i,:])
    point.set_ydata(y_list[i,:])
    point.set_3d_properties(z_list[i,:])

ani = FuncAnimation(fig=fig, func=animate, frames=N_MAX, interval=1, repeat=False, blit=False)
fig2 = plt.figure(figsize = (10,10))

#绘制表示x,y,z坐标的三个子图
ax2 = fig2.add_subplot(2,2,1)
ax3 = fig2.add_subplot(2,2,2)
ax4 = fig2.add_subplot(2,2,3)
#绘图
ax2.plot(x_list[:,:])
ax3.plot(y_list[:,:])
ax4.plot(z_list[:,:])
#给子图加标题
ax2.title.set_text("x")
ax3.title.set_text("y")
ax4.title.set_text("z")

plt.show()
