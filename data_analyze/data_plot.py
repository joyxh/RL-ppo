# coding=utf-8
from matplotlib import pyplot as plt
import matplotlib.tri as tri
import csv
import numpy as np
import pickle
from matplotlib import patches
import math
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm
import sys
sys.path.append('/root/pick/robot_learning/rllib/')
from envs.reacher_benchmark_env.reacher_benchmark import given_obj_pos, clip_j, NewRewF
import os

clip_obs = clip_j
NewRewF = NewRewF
sys.path.append('/root/pick/robot_learning/')

exp_id = '20181211-1818'
file_dir = os.path.join('experiments/ppo1/Reacher2BenchmarkEnv-v1/', exp_id) + '/'
# file_dir = '/home/pi/pick-place/experiments/ppo1/Reacher2BenchmarkEnv-v1/20181210-1106/'
# file_dir = '/home/pi/DQN/robot_learning/experiments/trpo_reset2/Reacher6BenchmarkEnv-v1/20181206-2217/'
dir_csv = file_dir + 'test_new.csv'
dir_pickle = file_dir + 'test.txt'
# dir_pickle = '/home/pi/DQN/robot_learning/experiments/trpo_reset2/Reacher6BenchmarkEnv-v1/20181205-1050/test.txt'
# dir_csv = 'test.csv'
if given_obj_pos:
    pic_name = str(given_obj_pos[0])+'_'+str(given_obj_pos[2])
else:
    pic_name = ''
csv_type = False


def strToList_int(x):
    pos_str = x.strip('[]')
    pos_arr = pos_str.split(',')
    for i in range(np.shape(pos_arr)[0]):
        pos_arr[i] = int(round(float(pos_arr[i]), 0))
    return pos_arr


def strToList_float(x):
    pos_str = x.strip('[]')
    pos_arr = pos_str.split(',')
    return np.array(pos_arr, dtype=float)


def read_csv():
    # read from csv
    target_pos, distance = [], []
    with open(dir_csv, 'r') as csvfile:
        dict_reader = csv.DictReader(csvfile)
        for row in dict_reader:
            target_pos.append(strToList_float(row['target_pos']))
            distance.append(float(row['distance']))
    return target_pos, distance


def read_pickle():
    with open(dir_pickle, 'rb') as file:
        b = pickle.load(file)
    # print(type(b['target_pos'][0][0]))
    mean_dic = {}
    for k, v in b.items():
        mean_dic[k] = np.mean(v)
    # print(mean_dic['target_pos'])
    return b['target_pos'], b['distance']


def contour_plot(real_x, real_z, distance, _cmap):
    # contour plot by interpolation
    triang = tri.Triangulation(real_x, real_z)
    interpolator = tri.LinearTriInterpolator(triang, distance)

    x_range = np.arange(min(real_x), max(real_x))
    z_range = np.arange(min(real_z), max(real_z))
    x_grid, z_grid = np.meshgrid(x_range, z_range)
    interpolated_distance = interpolator(x_grid, z_grid)

    # levels = np.arange(0, 100, 30)
    plt.contour(x_grid, z_grid, interpolated_distance, linewidths=3, cmap=_cmap, vmin=0.0)
    cs = plt.contour(x_grid, z_grid, interpolated_distance, linewidths=0.5, colors='black', vmin=0.0)
    plt.clabel(cs, fontsize=12, inline=True, inline_spacing=1, colors='black')

def plot_scatter():

    if csv_type:
        target_pos, distance = read_csv()
    else:
        target_pos, distance = read_pickle()
    target_pos = np.array(target_pos)
    distance = np.array(distance)

    real_x = target_pos[:, 0]
    real_z = target_pos[:, 2]
    plt.scatter(real_x, real_z, s=(max(distance)-np.array(distance))/30,
                c=distance, cmap='jet', edgecolors='none', vmin=0.0, vmax=100)
    plt.axvline(max(real_x), linestyle='--', linewidth=2)
    plt.axvline(min(real_x), linestyle='--', linewidth=2)
    plt.axhline(max(real_z), linestyle='--', linewidth=2)
    plt.axhline(min(real_z), linestyle='--', linewidth=2)

    plt.xlabel('x_axis[mm]')
    plt.ylabel('z_axis[mm]')
    plt.title(exp_id)
    plt.colorbar()
    file_name = './pictures/R2_old' + r'//' + exp_id + '.png'
    plt.savefig(file_name)
    plt.show()

def plot_2d_scatter():

    if csv_type:
        target_pos, distance = read_csv()
    else:
        target_pos, distance = read_pickle()

    target_pos = np.array(target_pos)
    distance = np.array(distance)

    real_x = target_pos[:, 0]
    real_z = target_pos[:, 2]

    # scatter plot
    _cmap = 'hsv'
    plt.scatter(real_x, real_z, s=(max(distance)-np.array(distance))/10,
                c=distance, cmap=_cmap, edgecolors='none', vmin=0.0, vmax=80)
    plt.colorbar()

    # contour_plot(real_x, real_z, distance, _cmap)
    # plot_circle()

    plt.axvline(max(real_x), linestyle='--', linewidth=2)
    plt.axvline(min(real_x), linestyle='--', linewidth=2)
    plt.axhline(max(real_z), linestyle='--', linewidth=2)
    plt.axhline(min(real_z), linestyle='--', linewidth=2)

    currentAxis = plt.gca()
    rect = patches.Rectangle((300, 200), 300, 300, linewidth=1, edgecolor='r', facecolor='none')
    currentAxis.add_patch(rect)

    plt.xlabel('x_axis[mm]')
    plt.ylabel('z_axis[mm]')
    plt.title(exp_id)
    file_name = './pictures/R2' + r'//' + exp_id + '.png'
    plt.savefig(file_name)
    plt.show()


def plot_3d_scatter():

    if csv_type:
        target_pos, distance = read_csv()
    else:
        target_pos, distance = read_pickle()
    target_pos = np.array(target_pos)
    distance = np.array(distance)

    real_x = target_pos[:, 0]
    real_y = target_pos[:, 1]
    real_z = target_pos[:, 2]

    fig = plt.figure(exp_id, facecolor='w')

    bx = [0, 0, 0, 0]
    bx[3] = fig.add_subplot(221, projection='3d')
    # scatter plot
    ddd = bx[3].scatter3D(real_x, real_y, real_z,s=(max(distance)-np.array(distance))/50, c=distance, edgecolors='none', cmap='jet', vmin=0.0, vmax=100)
    # fig.colorbar(ddd, ax=ax)

    bx[3].set_xlabel('x_axis[mm]')
    bx[3].set_ylabel('y_axis[mm]')
    bx[3].set_zlabel('z_axis[mm]')

    # bx[3].set_xticks((-800, 800, 200))
    # bx[3].set_yticks((-800, 800, 200))
    # bx[3].set_zticks(np.arange(-200, 1200, 200))

    xyz_type = 1    # 0:xy; 1:xz; 2:yz
    xyzd = [0, 0, 0]
    sub_num = [222, 223, 224]
    x_range = np.arange(301) + 300
    y_range = np.arange(401) - 200
    z_range = np.arange(301) + 200
    xyz_range = [x_range, y_range, z_range]
    # real_xyz = [real_x, real_y, real_z]
    real_xyz = [[real_x, real_y],
                [real_x, real_z],
                [real_y, real_z]]

    label_name = [['x_axis[mm]', 'y_axis[mm]'],
                  ['x_axis[mm]', 'z_axis[mm]'],
                  ['y_axis[mm]', 'z_axis[mm]']]

    # scatter plot
    for xyz_type in range(3):
        bx[xyz_type] = fig.add_subplot(sub_num[xyz_type])
        # scatter plot
        xyzd[xyz_type] = bx[xyz_type].scatter(real_xyz[xyz_type][0], real_xyz[xyz_type][1], s=(max(distance)-np.array(distance))/30,
                                              c=distance, cmap='jet', edgecolors='none', vmin=0.0, vmax=100)

        plt.axvline(max(real_xyz[xyz_type][0]), linestyle='--', linewidth=2)
        plt.axvline(min(real_xyz[xyz_type][0]), linestyle='--', linewidth=2)
        plt.axhline(max(real_xyz[xyz_type][1]), linestyle='--', linewidth=2)
        plt.axhline(min(real_xyz[xyz_type][1]), linestyle='--', linewidth=2)

        bx[xyz_type].set_xlabel(label_name[xyz_type][0])
        bx[xyz_type].set_ylabel(label_name[xyz_type][1])

        # plt.xticks(np.arange(-800, 800, 200))
        # plt.yticks(np.arange(-800, 800, 200))

    fig.suptitle(exp_id)
    fig.colorbar(xyzd[xyz_type], ax=bx)
    plt.show()

def plot_reward():
    # dx_m = obs[-3] * 0.001
    # dy_m = obs[-2] * 0.001
    # dz_m = obs[-1] * 0.001
    # distance_m = math.sqrt(dx_m ** 2 + dy_m ** 2 + dz_m ** 2)
    distance_m = np.arange(0, 1, 0.001)
    reward = -distance_m + np.exp(-100 * distance_m * distance_m)
    reward2 = -distance_m + np.exp(-200 * distance_m * distance_m)
    reward3 = -distance_m + np.exp(-50 * distance_m * distance_m)

    plt.plot(distance_m, reward, 'b', distance_m, reward2, 'r', distance_m, reward3, 'k')
    plt.xlabel("distance(m)")
    plt.ylabel("reward")
    plt.show()


def plot_circle():
    with open('experiments/circle.txt', 'rb') as file:
        b = pickle.load(file)
    b = np.array(b)
    cc_x = b[:, 0]
    cc_z = b[:, 2]
    plt.axvline(cc_x[np.argmax(cc_z)], color='k', linestyle='-.', linewidth=0.5)
    plt.axhline(cc_z[np.argmax(cc_x)], color='k',linestyle='-.', linewidth=0.5)
    plt.plot(cc_x, cc_z, 'k-', linewidth=3)

def plot_theta():
    _cmap = 'tab20b'
    with open(file_dir+'plot/'+pic_name+'reward.txt', 'rb') as file:
        b = pickle.load(file)
    theta = np.array(b['theta'])
    reward = np.array(b['rew'])
    # print(np.shape(theta1))
    # print(np.shape(theta2))
    # print(np.shape(reward))
    T1 = theta[:, 0]
    T2 = theta[:, 1]
    print(reward[5])
    # fig = plt.figure(figsize=(13,10))
    fig = plt.figure(figsize=(20,5))
    ax = fig.add_subplot(1, 1, 1)
    rew = ax.scatter(T1, T2, c=reward, cmap=_cmap, edgecolors='none', vmin=-1,vmax=1)
    fig.colorbar(rew, ax=ax)
    currentAxis = plt.gca()
    rect = patches.Rectangle((-5, -5), 10, 10, linewidth=1, edgecolor='r', facecolor='none')
    currentAxis.add_patch(rect)
    plot_action()
    plot_obs()
    plt.xlabel('theta1')
    plt.ylabel('theta2')
    # plt.xlim((-60, 135))
    # plt.ylim((-215, 45))
    plt.savefig(file_dir+'plot/'+pic_name+'.png')
    plt.show()

def plot_action():
    with open(file_dir+'plot/'+pic_name+'action.txt', 'rb') as file:
        a = pickle.load(file)
    print(a[0])
    theta = np.zeros((np.shape(a)[0]+1, np.shape(a)[1]))
    for i in range(np.shape(a)[0]+1):
        if i > 0:
            theta[i][0] = theta[i-1][0] + a[i-1][0]
            theta[i][1] = theta[i-1][1] + a[i-1][1]

    plt.plot(theta[:, 0], theta[:, 1], 'k', linewidth=1)
    # plt.show()

def plot_obs():
    with open('./' + file_dir+'plot/'+pic_name+'obs_clip.txt', 'rb') as file:
        obs_clip = pickle.load(file)
    with open('./' + file_dir+'plot/'+pic_name+'obs_no_clip.txt', 'rb') as file:
        obs_no_clip = pickle.load(file)

    obs_clip = np.array(obs_clip)
    # print(obs_clip)
    obs_no_clip = np.array(obs_no_clip)

    l=plt.plot(obs_clip[:, 0], obs_clip[:, 1], 'ko')
    plt.setp(l, markerfacecolor='r')
    plt.plot(obs_no_clip[:, 0], obs_no_clip[:, 1], 'b.', linewidth=0.1)
    # plt.show()

# def plot_var():
#     with open(file_dir+'plot/'+pic_name+'var.txt', 'rb') as file:
#         varinfo = pickle.load(file)
#     var = varinfo["var"]
#     act = varinfo["action"]
#     var = np.array(var)
#     act = np.array(act)
#
#     # print(np.shape(var[:,0,0]))
#     # print(np.shape(act[0]))
#     plt.plot(act[:,0, 0], var[:,0, 0], 'b.', linewidth=0.1)
#     plt.plot(act[:,0, 1], var[:,0, 1], 'r.', linewidth=0.1)
#
#     plt.show()


if __name__ == '__main__':
    # dis_xy = np.zeros([10, 3])
    # dis_xy = [[0 for i in range(10)] for j in range(3)]
    # a = [1,1,1,1,1,1,1,1,1,1]
    # x = dis_xy[2]
    # dis_xy[1]=a
    # dis_xy = np.array(dis_xy)
    # print(dis_xy[1,:])
    # plot_reward()

    # plot_scatter()

    # plot_2d_scatter()

    # plot_3d_scatter()

    # test()
    # read_pickle()
    plot_theta()
    # plot_action()
    # plot_obs()
    # plot_var()
