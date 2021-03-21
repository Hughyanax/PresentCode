# -*- coding: utf-8 -*-
"""
Created on Tue Feb 16 10:53:35 2021

@author: 86187
"""

import numpy as np
import math
from scipy.optimize import linear_sum_assignment
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d as p3
import matplotlib.animation as animation

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
delta_t = 0.1


#预设几个坐标
cross = [[0,0,0],[0,0,2],[0,0,4],[0,0,-2],[0,0,-4],[0,0,-6],[2,0,0],[4,0,0],[-2,0,0],[-4,0,0]]
triangle = [[0,0,0],[2,0,0],[-2,0,0],[1,0,2],[-1,0,2],[0,0,4],[1,0,-2],[3,0,-2],[-1,0,-2],[-3,0,-2]]
dodecahedron = [[0,0,0],[1,1,-2],[1,-1,-2],[-1,1,-2],[-1,-1,-2],[1,1,-4],[1,-1,-4],[-1,1,-4],[-1,-1,-4],[0,0,-6]]
#确定各无人机初始位置
initial_shape = cross
#确定一个目标队形
shape = dodecahedron


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
row_ind,col_ind = linear_sum_assignment(cost)
for i in range(NUM):
    follower[i].confirm_aim(shape[col_ind[i]])


#为了方便简化表示邻接矩阵
omega = adjacent_matrix 
#初始化各无人机加速度
accelerate_speed = [np.zeros(3)] * NUM
#简化表示加速度
alpha = accelerate_speed
#初始化各无人机方向角cos值
cos_rad = [np.zeros(3)] * NUM
#确定论文中参数gamma的值
gamma = 1.5

location = initial_shape
aim_location = shape
for i in range(NUM):
    aim_location[i] = np.array(follower[i].aim_pot)
    location[i] = np.array([follower[i].x, follower[i].y, follower[i].z])
    
k = 1
while k <= 100 :
    
    x_list.append([])
    y_list.append([])
    z_list.append([])
    for i in range(NUM):
        location[i] = np.array([follower[i].x, follower[i].y, follower[i].z])
        follower[i].rad = follower[i].angel_calculate(aim_location[i])
        alpha[i][0] = 0
        alpha[i][1] = 0
        alpha[i][2] = 0
        for j in range(NUM):
            follower[j].rad = follower[j].angel_calculate(aim_location[j])
            #下式为论文式（3）
            alpha[i][0] = alpha[i][0] - omega[i][j] * (((location[i][0] - aim_location[i][0]) - (location[j][0] - aim_location[j][0])) + gamma * (follower[i].vec * math.cos(follower[i].rad[0])- follower[j].vec * math.cos(follower[j].rad[0])))
            alpha[i][1] = alpha[i][1] - omega[i][j] * (((location[i][1] - aim_location[i][1]) - (location[j][1] - aim_location[j][1])) + gamma * (follower[i].vec * math.cos(follower[i].rad[1])- follower[j].vec * math.cos(follower[j].rad[1])))
            alpha[i][2] = alpha[i][2] - omega[i][j] * (((location[i][2] - aim_location[i][2]) - (location[j][2] - aim_location[j][2])) + gamma * (follower[i].vec * math.cos(follower[i].rad[2])- follower[j].vec * math.cos(follower[j].rad[2])))
        follower[i].x = follower[i].x + follower[i].vec * follower[i].rad[0] * delta_t + 0.5 * alpha[i][0] * delta_t**2
        follower[i].y = follower[i].y + follower[i].vec * follower[i].rad[1] * delta_t + 0.5 * alpha[i][1] * delta_t**2
        follower[i].z = follower[i].z + follower[i].vec * follower[i].rad[2] * delta_t + 0.5 * alpha[i][2] * delta_t**2
        follower[i].vec = ((follower[i].vec * math.cos(follower[i].rad[0]) + alpha[i][0] * delta_t)**2 +
                            (follower[i].vec * math.cos(follower[i].rad[1]) + alpha[i][1] * delta_t)**2 + 
                            (follower[i].vec * math.cos(follower[i].rad[0]) + alpha[i][2] * delta_t)**2)**0.5
        x_list[k].append(follower[i].x)
        y_list[k].append(follower[i].y)
        z_list[k].append(follower[i].z)
    k = k + 1
        

print(y_list)
                
    


















