# -*- coding: utf-8 -*-
"""
Created on Tue Feb 16 10:53:35 2021

@author: 86187
"""

import numpy as np
import math
from scipy.optimize import linear_sum_assignment
import matplotlib.pyplot as plt
import matplotlib.animation as animation
#import mpl_toolkits.mplot3d as p3

'''
10个无人机的邻接矩阵，规范同论文一致,leader为首行首列
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


#设置无人机数目
NUM = 10
#确定最小时间间隔,单位是秒
delta_t = 0.2
#确定循环次数times
circul_times = 50

#预设几个坐标
#浮点型
#cross = [[0.0,0.0,0.0],[0.0,0.0,2.0],[0.0,0.0,4.0],[0.0,0.0,-2.0],[0.0,0.0,-4.0],[0.0,0.0,-6.0],[2.0,0.0,0.0],[4.0,0.0,0.0],[-2.0,0.0,0.0],[-4.0,0.0,0.0]]
#triangle = [[0,0,0],[2,0,0],[-2,0,0],[1,0,2],[-1,0,2],[0,0,4],[1,0,-2],[3,0,-2],[-1,0,-2],[-3,0,-2]]
#dodecahedron = [[0.0,0.0,0.0],[1.0,1.0,-2.0],[1.0,-1.0,-2.0],[-1.0,1.0,-2.0],[-1.0,-1.0,-2.0],[1.0,1.0,-4.0],[1.0,-1.0,-4.0],[-1.0,1.0,-4.0],[-1.0,-1.0,-4.0],[0.0,0.0,-6.0]]

#整型
cross = [[0,0,0],[0,0,2],[0,0,4],[0,0,-2],[0,0,-4],[0,0,-6],[2,0,0],[4,0,0],[-2,0,0],[-4,0,0]]
triangle = [[0,0,0],[2,0,0],[-2,0,0],[1,0,2],[-1,0,2],[0,0,4],[1,0,-2],[3,0,-2],[-1,0,-2],[-3,0,-2]]
dodecahedron = [[0,0,0],[1,1,-2],[1,-1,-2],[-1,1,-2],[-1,-1,-2],[1,1,-4],[1,-1,-4],[-1,1,-4],[-1,-1,-4],[0,0,-6]]

#确定各无人机初始位置
initial_shape = cross
#确定一个目标队形
shape = triangle

'''
x_list = []
y_list = []
z_list = []
while True:
    x_list.append([])
    y_list.append([])
    z_list.append([])
    for i in range(NUM):
        x_list[0].append(initial_shape[i][0])
        y_list[0].append(initial_shape[i][1])
        z_list[0].append(initial_shape[i][2])
    break
'''
x_list = np.ndarray([circul_times, NUM])
y_list = np.ndarray([circul_times, NUM])
z_list = np.ndarray([circul_times, NUM])
for i in range(NUM):
    x_list[0][i] = initial_shape[i][0]
    y_list[0][i] = initial_shape[i][1]
    z_list[0][i] = initial_shape[i][2]

class Leader():
    '''
    初始化无人机的坐标[x,y,z];速度vec;航迹方位角rad
    '''
    def __init__(self, x, y, z, vec, rad):
        self.x = x
        self.y = y
        self.z = z
        self.aim_pot = []
        if np.shape(rad) != (3,):
            print('rad input error')
        self.vec = vec
        self.rad = rad
        
    def confirm_aim(self, aim):
        '''
        确定目标
        '''
        self.aim_pot = aim
        return self.aim_pot
    
    def move(self):
        '''
        让无人机朝指定方向移动
        '''
        self.x = self.x + math.cos(self.rad[0]) * self.vec
        self.y = self.y + math.cos(self.rad[1]) * self.vec
        self.z = self.z + math.cos(self.rad[2]) * self.vec
    
    def distance_calculate(self, aim):
        '''
        计算该无人机与目标之间的距离,返回一个浮点数
        '''
        delta_x = aim[0] - self.x
        delta_y = aim[1] - self.y
        delta_z = aim[2] - self.z
        distance = (delta_x**2 + delta_y**2 + delta_z**2)**0.5
        return distance
    
    def angel_calculate(self, aim):
        '''
        计算该无人机与目标之间的夹角
        '''
        distance = self.distance_calculate(aim)
        if distance > 0:
            theta_x = math.acos((aim[0] - self.x) / distance)
            theta_y = math.acos((aim[1] - self.y) / distance)
            theta_z = math.acos((aim[2] - self.z) / distance)
            theta = [theta_x, theta_y, theta_z]
        else:
            theta = [0, 0, 0]
        return theta


follower = [Leader(x_list[0][i], y_list[0][i], z_list[0][i],0,[0,0,0]) for i in range(NUM)]


#确定邻接矩阵
adjacent_matrix = np.zeros([NUM,NUM])
for i in range(NUM):
    for j in range(NUM):
        if j == i - 1 or j == i - 2:
            adjacent_matrix[i][j] = 1
#生成各无人机与目标点之间距离的cost矩阵
cost = np.zeros([NUM, NUM])
for i in range(NUM):
    for j in range(NUM):
        cost[i][j] = follower[i].distance_calculate(shape[j])
#匈牙利算法确定各无人机目标的最优解,并以此确定各无人机目标坐标
row_ind, col_ind = linear_sum_assignment(cost)
for i in range(NUM):
    follower[i].confirm_aim(shape[col_ind[i]])


#为了方便简化表示邻接矩阵
omega = adjacent_matrix
#初始化度矩阵
degree_matrix = np.diag([2,2,2,2,2,2,2,2,1,0])
#初始化各无人机方向角cos值
cos_rad = [np.zeros(3)] * NUM
#确定论文中参数gamma的值
gamma = 1.6
#确定laplace矩阵
lapalce = degree_matrix - omega
#预先计算论文式(6)中的两项张量积
I_m = np.array([[1],[1],[1]])
first_item = np.kron(-lapalce, I_m)
second_item = np.kron(lapalce, I_m)
#论文中c矩阵
c_matrix = np.array([])

location = np.array(initial_shape)
aim_location = np.array(shape)
miu = np.array(shape)

for i in range(NUM):
    aim_location[i] = follower[i].aim_pot
    location[i][0] = follower[i].x
    location[i][1] = follower[i].y
    location[i][2] = follower[i].z
    miu[i] = [follower[i].vec * math.cos(follower[i].rad[0]), follower[i].vec * math.cos(follower[i].rad[1]), follower[i].vec * math.cos(follower[i].rad[2])]


k = 1
while k < circul_times:
    c_matrix = np.dot(first_item, (location - aim_location)) - gamma * np.dot(second_item, miu)
    for i in range(NUM):
        follower[i].rad = follower[i].angel_calculate(aim_location[i])
        follower[i].x = follower[i].x + follower[i].vec * math.cos(follower[i].rad[0]) * delta_t + 0.5 * c_matrix[3 * i][0] * delta_t**2
        follower[i].y = follower[i].y + follower[i].vec * math.cos(follower[i].rad[1]) * delta_t + 0.5 * c_matrix[3 * i][1] * delta_t**2
        follower[i].z = follower[i].z + follower[i].vec * math.cos(follower[i].rad[2]) * delta_t + 0.5 * c_matrix[3 * i][2] * delta_t**2
        x_list[k][i] = follower[i].x
        y_list[k][i] = follower[i].y
        z_list[k][i] = follower[i].z
        location[i][0] = follower[i].x
        location[i][1] = follower[i].y
        location[i][2] = follower[i].z
        miu[i] = [follower[i].vec * math.cos(follower[i].rad[0]) + c_matrix[3 * i][0] * delta_t, follower[i].vec * math.cos(follower[i].rad[1]) + c_matrix[3 * i][1] * delta_t, follower[i].vec * math.cos(follower[i].rad[2]) + c_matrix[3 * i][2] * delta_t]
        follower[i].vec = ((miu[i][0])**2 + (miu[i][1])**2 + (miu[i][2])**2)**0.5
    k = k + 1

fig  = plt.figure(figsize=(10,10))
ax = fig.add_subplot(1,1,1,projection='3d')
point, = ax.plot(x_list[0][:],y_list[0][:],z_list[0][:],'r.', marker = '1')
ax.set_xlim3d([-10,10])
ax.set_xlabel('x')
ax.set_ylim3d([-10,10])
ax.set_ylabel('y')
ax.set_zlim3d([-10,10])
ax.set_zlabel('z')

def animate(i):
    point.set_xdata(x_list[i][:])
    point.set_ydata(y_list[i][:])
    point.set_3d_properties(z_list[i][:])

ani = animation.FuncAnimation(fig=fig, func=animate, frames=k, interval=1, repeat=False, blit=False)


fig2 = plt.figure(figsize = (10,10))
#绘制表示x,y,z坐标的三个子图
ax2 = fig2.add_subplot(2,2,1)
ax3 = fig2.add_subplot(2,2,2)
ax4 = fig2.add_subplot(2,2,3)
#绘图
ax2.plot(x_list[:][:])
ax3.plot(y_list[:][:])
ax4.plot(z_list[:][:])
#给子图加标题
ax2.title.set_text("x")
ax3.title.set_text("y")
ax4.title.set_text("z")


plt.show()
