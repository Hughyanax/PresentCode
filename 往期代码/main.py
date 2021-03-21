'''忽略飞行器姿态以及碰撞，忽略leader-follower，初步编队仿真尝试'''

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment
from matplotlib.animation import FuncAnimation
import time
import sys
import copy

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

#随机生成无人机初始坐标
#x_initial = np.random.randint(-6,6,10)
#y_initial = np.random.randint(-6,6,10)
#z_initial = np.random.randint(-6,6,10)
x_initial = [0,0,0,0,0,0,0,0,0,0]
y_initial = [0,1,2,-1,-2,0,0,0,0,0]
z_initial = [0,0,0,0,0,1,2,-1,-2,-3]

'''预设几个队列坐标'''
#十二面体坐标
dodecahedron = [[0,0,0],[1,1,-2],[1,-1,-2],[-1,1,-2],[-1,-1,-2],[1,1,-4],[1,-1,-4],[-1,1,-4],[-1,-1,-4],[0,0,-6]]
#T形
T_shape = [[0,0,0],[0,1,0],[0,2,0],[0,-1,0],[0,-2,0],[0,0,1],[0,0,2],[0,0,-1],[0,0,-2],[0,0,-3]]

#选择预设坐标
shape = dodecahedron

#储存各无人机当前坐标
pot_list = []
for i in range(NUM):
    pot_list.append([x_initial[i],y_initial[i],z_initial[i]])

class follower():
    MOVE_DISTANCE = MOVE_DISTANCE
    aim_pot = None
    
    def __init__(self,id):
        self.id = id
        self.location = [x_initial[id],y_initial[id],z_initial[id]]

    def distance_calculate(self,shape):
        '''计算该无人机与各目标点间的距离,返回距离的一维数组'''
        location = self.location
        distance = []
        for each in shape:
            Distance = ((location[0] - each[0])**2 + (location[1] - each[1])**2 + (location[2] - each[2])**2)**0.5
            distance.append(Distance)
        return distance

    def aim_distance(self,aim):
        '''计算该无人机与目标坐标之间的距离，返回一个float距离'''
        location = self.location
        distance = ((location[0] - aim[0])**2 + (location[1] - aim[1])**2 + (location[2] - aim[2])**2)**0.5
        return distance

    def aim_location(self,shape,col_ind):
        '''通过匈牙利算法结果得到目标坐标,返回目标坐标（三维）'''
        aim_location = shape[col_ind[self.id]]
        return aim_location

    def angle_calculate(self,aim):
        '''计算起始坐标与目标坐标形成直线在x,y,z轴上夹角的cos值，返回一个三维列表'''
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
cost = np.zeros([10,10])
for i in range(NUM):
    for j in range(NUM):
        trans = obj_list[i].distance_calculate(shape)
        cost[i][j] = trans[j]

#匈牙利算法分配路径,确定目标坐标
row_ind,col_ind = linear_sum_assignment(cost)
for i in range(NUM):
    obj_list[i].aim_pot = obj_list[i].aim_location(shape,col_ind)

fig, ax = plt.subplots()
ax.set_xlim(-10,10)
ax.set_ylim(-10,10)
sc = ax.scatter(x_initial,y_initial,z_initial,color = 'r', alpha = 0.7, marker = '1', linewidth = 10)

def update(N):
    li = [0,0,0,0,0,0,0,0,0,0]
    comp_li = [1,1,1,1,1,1,1,1,1,1]
    while True:
        for i in range(NUM):
            aim = obj_list[i].aim_pot
            obj_list[i].move(aim)
            pot_list[i] = obj_list[i].location
            if obj_list[i].location == aim:
                li[i] = 1
        if li == comp_li:
            time.sleep(3)
            plt.close()
            sys.exit()
        sx,sy,sz = [],[],[]
        for each in pot_list:
            sx.append(each[0])
            sy.append(each[1])
            sz.append(each[2])
            sc.set_offsets(np.c_[sx, sy, sz])
        return sc

ani = FuncAnimation(fig, update, frames =  np.linspace(1,50), interval = 1)
plt.show()
    
    
        
            
            
            





















