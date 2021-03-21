# -*- coding: utf-8 -*-

"""
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
[1,0,0,0,0,0,0,1,1,0]

度矩阵,仅考虑入度，leader为首行首列
B = diag([0,1,2,2,2,2,2,2,2,2])

L = B - A
-L特征值为：
diag([2,2,2,2,2,2,2,2,1,0])
"""


import numpy as np
import math
from scipy.optimize import linear_sum_assignment
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
import copy

# 设置无人机数目
NUM = 10
# 确定最小时间间隔,单位是秒
delta_t = float(1 / 60)
# 确定仿真总时间/s
total_t = 4
# 设置无人机最大加速度
max_accelerate = 8.0
# 设置循环次数
circle_times = int(total_t / delta_t)
# 安全距离
d = 3
# 避碰参数设置
h_1 = np.array([1, 0, 0])
h_2 = np.array([0, 1, 0])
k_p = float(0.5)
l_p = float(0.3)
# 确定邻接矩阵
adjacent_matrix = np.zeros([NUM, NUM])
for i in range(NUM):
    for j in range(NUM):
        if j == i - 1 or j == i - 2:
            adjacent_matrix[i][j] = 1
# 简化表示邻接矩阵
omega = adjacent_matrix
# 初始化度矩阵
degree_matrix = np.diag([0, 1, 2, 2, 2, 2, 2, 2, 2, 2])
# 确定论文中参数gamma的值(负值仿真可行，论文式六存疑)
gamma = -5.0
# 确定laplace矩阵
laplace = degree_matrix - omega
# 预先计算论文式(6)中的两项张量积
I_m = np.array([[1.0], [1.0], [1.0]])
first_item = np.kron(-laplace, I_m)
second_item = np.kron(laplace, I_m)
# 论文中c矩阵
c_matrix = np.array([])
# 预设几个坐标
cross = [[0.0, 0.0, 0.0], [0.0, 0.0, 2.0], [0.0, 0.0, 4.0], [0.0, 0.0, -2.0], [0.0, 0.0, -4.0], [0.0, 0.0, -6.0],
         [2.0, 0.0, 0.0], [4.0, 0.0, 0.0], [-2.0, 0.0, 0.0], [-4.0, 0.0, 0.0]]
triangle = [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [-2.0, 0.0, 0.0], [1.0, 0.0, 2.0], [-1.0, 0.0, 2.0], [0.0, 0.0, 4.0],
        [1.0, 0.0, -2.0], [3.0, 0.0, -2.0], [-1.0, 0.0, -2.0],[-3.0, 0.0, -2.0]]
dodecahedron = [[0.0, 0.0, 0.0], [1.0, 1.0, -2.0], [1.0, -1.0, -2.0], [-1.0, 1.0, -2.0], [-1.0, -1.0, -2.0],
                [1.0, 1.0, -4.0], [1.0, -1.0, -4.0], [-1.0, 1.0, -4.0], [-1.0, -1.0, -4.0], [0.0, 0.0, -6.0]]
initialize = np.zeros([10,3])
# 确定各无人机初始位置
initial_shape = triangle
# 确定一个目标队形
shape = copy.deepcopy(triangle)
# shape = copy.deepcopy(dodecahedron)
# 随机生成坐标
x_initial = np.random.randint(-6,6,10)
y_initial = np.random.randint(-6,6,10)
z_initial = np.random.randint(-6,6,10)

# 按initial_shape得到初始坐标
# x_initial = [initial_shape[i][0] + 2 for i in range(NUM)]
# y_initial = [initial_shape[i][1] + 2 for i in range(NUM)]
# z_initial = [initial_shape[i][2] + 2 for i in range(NUM)]

x_list = np.ndarray([circle_times, NUM])
y_list = np.ndarray([circle_times, NUM])
z_list = np.ndarray([circle_times, NUM])
for i in range(NUM):
    x_list[0][i] = x_initial[i]
    y_list[0][i] = y_initial[i]
    z_list[0][i] = z_initial[i]
for i in range(NUM):
    shape[i][0] = x_initial[0] + shape[i][0]
    shape[i][1] = y_initial[0] + shape[i][1]
    shape[i][2] = z_initial[0] + shape[i][2]
# 计算两点之间距离
def distance(location1, location2):
    sum_val = 0
    for i in range(3):
        sum_val += (location1[i] - location2[i]) ** 2
    return math.sqrt(sum_val)


class Leader:
    """初始化无人机的坐标[x,y,z];速度vec;航迹方位角rad"""
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
        """确定目标"""
        self.aim_pot = aim
        return self.aim_pot

    def distance_calculate(self, aim):
        """计算该无人机与目标之间的距离,返回一个浮点数"""
        delta_x = aim[0] - self.x
        delta_y = aim[1] - self.y
        delta_z = aim[2] - self.z
        return math.sqrt(delta_x ** 2 + delta_y ** 2 + delta_z ** 2)

    def angel_calculate(self, aim):
        """计算该无人机与目标之间的夹角"""
        if self.distance_calculate(aim) > 0:
            theta_x = math.acos((aim[0] - self.x) / self.distance_calculate(aim))
            theta_y = math.acos((aim[1] - self.y) / self.distance_calculate(aim))
            theta_z = math.acos((aim[2] - self.z) / self.distance_calculate(aim))
            theta = [theta_x, theta_y, theta_z]
        else:
            theta = [0, 0, 0]
        return theta


class Follower(Leader):
    def __init__(self, x, y, z, vec, rad):
        super().__init__(x, y, z, vec, rad)


UAVs = [Leader(x_list[0][0], y_list[0][0], z_list[0][0], 0, [0, 0, 0])] + [
    Follower(x_list[0][i], y_list[0][i], z_list[0][i], 0, [0, 0, 0]) for i in range(1, NUM, 1)]


# 生成各无人机与目标点之间距离的cost矩阵
cost = np.zeros([NUM, NUM])
for i in range(NUM):
    for j in range(NUM):
        cost[i][j] = UAVs[i].distance_calculate(shape[j])
# 匈牙利算法确定各无人机目标的最优解,并以此确定各无人机目标坐标
row_ind, col_ind = linear_sum_assignment(cost)
new_shape = np.zeros([NUM,3], dtype = float)
for i in range(NUM):
    UAVs[i].confirm_aim(shape[col_ind[i]])
    new_shape[i] = [shape[col_ind[i]][0] - UAVs[0].x, shape[col_ind[i]][1] - UAVs[0].y, shape[col_ind[i]][2] - UAVs[0].z]

# 初始化各无人机方向角cos值
cos_rad = [np.zeros(3)] * NUM

location = np.array(initial_shape, dtype=float)
aim_location = np.array(shape, dtype=float)
miu = np.zeros([10,3],dtype=float)
# 采集数据
x_vec_list = np.zeros([circle_times,NUM])
y_vec_list = np.zeros([circle_times,NUM])
z_vec_list = np.zeros([circle_times,NUM])
x_acce_list = np.zeros([circle_times,NUM])
y_acce_list = np.zeros([circle_times,NUM])
z_acce_list = np.zeros([circle_times,NUM])
for i in range(NUM):
    aim_location[i] = UAVs[i].aim_pot
    location[i][0] = UAVs[i].x
    location[i][1] = UAVs[i].y
    location[i][2] = UAVs[i].z
    miu[i, 0] = UAVs[i].vec * math.cos(UAVs[i].rad[0])
    miu[i, 1] = UAVs[i].vec * math.cos(UAVs[i].rad[1])
    miu[i, 2] = UAVs[i].vec * math.cos(UAVs[i].rad[2])


k = 1
while k < circle_times:
    leader_location = np.array([location[0][0], location[0][1], location[0][2]])
    I_NUM = np.array([[1], [1], [1], [1], [1], [1], [1], [1], [1], [1]])
    leader_vec = np.array([miu[0][0], miu[0][1], miu[0][2]])
    c_matrix = np.dot(first_item, (location - (I_NUM * leader_location + new_shape))) - gamma * np.dot(second_item,miu - I_NUM * leader_vec)
    for i in range(NUM):
        # 避碰,调整加速度
        adjacent_NUM = 0
        for j in range(NUM):
            if distance([UAVs[i].x, UAVs[i].y, UAVs[i].z], [UAVs[j].x, UAVs[j].y, UAVs[j].z]) <= d:
                adjacent_NUM += 1
        p = 0
        while p <= adjacent_NUM:
            r_i = np.array([UAVs[i].x - UAVs[p].x, UAVs[i].y - UAVs[p].y, UAVs[i].z - UAVs[p].z], dtype=float)
            if np.dot(r_i, h_1) < np.dot(r_i, h_2):
                c1 = np.cross(k_p * r_i, h_1)
                c_matrix[3 * i] += np.cross(k_p * r_i, h_1) * l_p
            else:
                c2 = np.cross(k_p * r_i, h_2)
                c_matrix[3 * i] += np.cross(k_p * r_i, h_2) * l_p
            p += 1

        # 加速度限制
        for m in range(3):
            if (c_matrix[3 * i][m] < -max_accelerate):
                c_matrix[3 * i][m] = -max_accelerate
            elif (c_matrix[3 * i][m] > max_accelerate):
                c_matrix[3 * i][m] = max_accelerate
                

        # 更新位置
        if i == 0:
            # 为leader设置路线(匀速运动)
            leader_x_vec = 0
            leader_y_vec = 0
            leader_z_vec = 0
            c_matrix[i] = [0, 0, 0]
            UAVs[i].x = UAVs[i].x + leader_x_vec * delta_t
            UAVs[i].y = UAVs[i].y + leader_y_vec * delta_t
            UAVs[i].z = UAVs[i].z + leader_z_vec * delta_t
            miu[i] = [leader_x_vec, leader_y_vec, leader_z_vec]
            UAVs[i].vec = ((miu[i][0]) ** 2 + (miu[i][1]) ** 2 + (miu[i][2]) ** 2) ** 0.5
            x_vec_list[k][i] = miu[i][0]
            y_vec_list[k][i] = miu[i][1]
            z_vec_list[k][i] = miu[i][2]
            x_acce_list[k][i] = c_matrix[3 * i][0]
            y_acce_list[k][i] = c_matrix[3 * i][1]
            z_acce_list[k][i] = c_matrix[3 * i][2]
        else:
            UAVs[i].rad = UAVs[i].angel_calculate(new_shape[i] + leader_location)
            UAVs[i].x = (UAVs[i].x + UAVs[i].vec * math.cos(UAVs[i].rad[0]) * delta_t + 
                        0.5 * c_matrix[3 * i][0] * delta_t ** 2)
            UAVs[i].y = (UAVs[i].y + UAVs[i].vec * math.cos(UAVs[i].rad[1]) * delta_t + 
                        0.5 * c_matrix[3 * i][1] * delta_t ** 2)
            UAVs[i].z = (UAVs[i].z + UAVs[i].vec * math.cos(UAVs[i].rad[2]) * delta_t + 
                        0.5 * c_matrix[3 * i][2] * delta_t ** 2)
            miu[i] = [UAVs[i].vec * math.cos(UAVs[i].rad[0]) + c_matrix[3 * i][0] * delta_t,
                      UAVs[i].vec * math.cos(UAVs[i].rad[1]) + c_matrix[3 * i][1] * delta_t,
                      UAVs[i].vec * math.cos(UAVs[i].rad[2]) + c_matrix[3 * i][2] * delta_t]
        x_list[k][i] = UAVs[i].x
        y_list[k][i] = UAVs[i].y
        z_list[k][i] = UAVs[i].z
        location[i][0] = UAVs[i].x
        location[i][1] = UAVs[i].y
        location[i][2] = UAVs[i].z
        UAVs[i].vec = ((miu[i][0]) ** 2 + (miu[i][1]) ** 2 + (miu[i][2]) ** 2) ** 0.5
        x_vec_list[k][i] = miu[i][0]
        y_vec_list[k][i] = miu[i][1]
        z_vec_list[k][i] = miu[i][2]
        x_acce_list[k][i] = c_matrix[3 * i][0]
        y_acce_list[k][i] = c_matrix[3 * i][1]
        z_acce_list[k][i] = c_matrix[3 * i][2]
    k = k + 1


#绘制表示x,y,z坐标的三个子图
fig2 = plt.figure(figsize = (16,9))
ax2 = fig2.add_subplot(3,3,1)
ax3 = fig2.add_subplot(3,3,2)
ax4 = fig2.add_subplot(3,3,3)
ax5 = fig2.add_subplot(3,3,4)
ax6 = fig2.add_subplot(3,3,5)
ax7 = fig2.add_subplot(3,3,6)
ax8 = fig2.add_subplot(3,3,7)
ax9 = fig2.add_subplot(3,3,8)
ax10 = fig2.add_subplot(3,3,9)
#绘图
ax2.plot(x_list[:][:])
ax3.plot(y_list[:][:])
ax4.plot(z_list[:][:])
ax5.plot(x_vec_list[:][:])
ax6.plot(y_vec_list[:][:])
ax7.plot(z_vec_list[:][:])
ax8.plot(x_acce_list[:][:])
ax9.plot(y_acce_list[:][:])
ax10.plot(z_acce_list[:][:])
#给子图加标题
ax2.title.set_text("x")
ax3.title.set_text("y")
ax4.title.set_text("z")
ax5.title.set_text("x_vec")
ax6.title.set_text("y_vec")
ax7.title.set_text("z_vec")
ax8.title.set_text("x_acce")
ax9.title.set_text("y_acce")
ax10.title.set_text("z_acce")
plt.savefig("outputpicture.png")

# 将移动路线保存为MP4
metadata = dict(title='Movie Test', artist='Matplotlib', comment='Movie support!')
writer = FFMpegWriter(fps=30, metadata=metadata)
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(1, 1, 1, projection='3d')
#限制坐标轴范围,并加上标签
ax.set_xlim3d([-10,10])
ax.set_xlabel('x')
ax.set_ylim3d([-10,10])
ax.set_ylabel('y')
ax.set_zlim3d([-10,10])
ax.set_zlabel('z')
point, = ax.plot(x_list[0][:],y_list[0][:],z_list[0][:],'r.', marker = '1')
with writer.saving(fig, "writer_test.mp4", 100):  # 100指的dpi，dot per inch，表示清晰度
    for i in range(circle_times):
        point.set_xdata(x_list[i][:])
        point.set_ydata(y_list[i][:])
        point.set_3d_properties(z_list[i][:])
        plt.pause(0.001)
        writer.grab_frame()
