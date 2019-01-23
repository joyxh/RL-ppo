# coding=utf-8
from matplotlib import pyplot as plt
import matplotlib.tri as tri
import csv
import numpy as np
import pickle
import matplotlib.patches as patches
import matplotlib as mpl
import argparse
import math
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm
import os
import os.path as osp
import sys
sys.path.append("./rllib/")
from envs.reacher_benchmark_env.reacher_benchmark import clip_j
# import envs.reacher_benchmark_env.reacher_benchmark.ReacherBenchmarkEnv._get_end_effector_pos as eep
from data_analyze.plot_hyperparameter_ppo1 import LOG_DIR, env_id

clip_obs = clip_j

# # 一次画LOG_DIR文件夹下所有数据的图，调用plot_hyperparameter_ppo1.py实现
# parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
# parser.add_argument('--seed', help='random seed', default=4)
# parser.add_argument('--exp-id', help='experiment id', default=None)
# parser.add_argument('--test-id', help='test id', default=None)
# parser.add_argument('--env-id', help='environment id', default='ReacherBenchmarkEnv-v1')
# args = parser.parse_args()
# exp_id = args.exp_id
# test_id = args.test_id
# file_dir = LOG_DIR + '/' + exp_id + '/test0/TagetScatter/Random/' + test_id + '/'

# 一次只画一个图，针对有多个测试数据的情形，需手动输入test_id
exp_id = '20190116-1744'
test_id = '20190118-1453'
file_dir = 'experiments/ppo1_R6/ReacherBenchmarkEnv-v1/' + exp_id + '/test0/JAreward/Random/' + test_id + '/'

# # 一次只画一个图，针对file_dir_下只有一个测试数据文件的情况
# exp_id = '20190121-1305'
# file_dir = 'experiments/ppo1_R6/ReacherBenchmarkEnv-v1/'+exp_id+'/test0/JAreward/AllPos/'
# test_list = os.listdir(file_dir)
# file_dir = file_dir + test_list[-1] + '/'

dir_csv = file_dir + 'test_new.csv'
dir_pickle = file_dir + 'test.txt'
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


def read_csv(dir_csv_):
    # read from csv
    target_pos, distance, episode_rew = [], [], []
    with open(dir_csv_, 'r') as csvfile:
        dict_reader = csv.DictReader(csvfile)
        for row in dict_reader:
            target_pos.append(strToList_float(row['target_pos']))
            distance.append(float(row['distance']))
            episode_rew.append(float(row['episode_rew']))
    return target_pos, distance, episode_rew


def read_pickle_mean(dir_pickle_):
    with open(dir_pickle_, 'rb') as file:
        b = pickle.load(file)
    # print(type(b['target_pos'][0][0]))
    mean_dic = {}
    for k, v in b.items():
        mean_dic[k] = np.mean(v)
    # print(mean_dic['target_pos'])
    return b['target_pos'], b['distance'], b['episode_rew']


def read_pickle_plot(pic_name, file_name):
    with open(file_dir+'plot/'+pic_name+file_name, 'rb') as file:
        a = pickle.load(file)
    return a


def read_data(exp_ids):
    file_dirs, dir_csvs, dir_pickles = [], [], []
    for exp_id_ in exp_ids:
        file_dirs_, dir_csvs_, dir_pickles_ = [], [], []
        for expID in exp_id_:
            file_dir_ = os.path.join(LOG_DIR, env_id) + '/' + expID + '/test/TagetScatter/Random/'
            test_list = os.listdir(file_dir_)
            # 针对file_dir_下只有一个测试文件的情况
            test_dir_ = file_dir_ + test_list[-1]
            dir_csv_ = test_dir_ + '/test_new.csv'
            dir_pickle_ = test_dir_ + '/test.txt'
            # file_dirs_.append(file_dir)
            dir_csvs_.append(dir_csv_)
            dir_pickles_.append(dir_pickle_)
        dir_csvs.append(dir_csvs_)
        dir_pickles.append(dir_pickles_)
    target_poses, distances, episode_rews = [], [], []
    if csv_type:
        for dir_csvs_ in dir_csvs:
            target_poses_, distances_, episode_rews_ = [], [], []
            for dir_csv_ in dir_csvs_:
                target_pos, distance, episode_rew = read_csv(dir_csv_)
                target_poses_ = target_poses_ + target_pos
                distances_ = distances_ + distance
                episode_rews_ = episode_rews_ + episode_rew
            target_poses.append(target_poses_)
            distances.append(distances_)
            episode_rews.append(episode_rews_)
    else:
        for dir_pickles_ in dir_pickles:
            target_poses_, distances_, episode_rews_ = [], [], []
            for dir_pickle_ in dir_pickles_:
                target_pos, distance, episode_rew = read_pickle_mean(dir_pickle_)
                target_poses_ = target_poses_ + target_pos
                distances_ = distances_ + distance
                episode_rews_ = episode_rews_ + episode_rew
            target_poses.append(target_poses_)
            distances.append(distances_)
            episode_rews.append(episode_rews_)

    distances = np.array(distances)
    distances = np.transpose(distances)
    episode_rews = np.array(episode_rews)
    episode_rews = np.transpose(episode_rews)
    print('******Data_size = ', distances.shape)

    return distances, episode_rews


def plot_range_box():
    currentAxis = plt.gca()
    rect = patches.Rectangle((300, 200), 300, 300, linewidth=1, edgecolor='r', facecolor='none')
    currentAxis.add_patch(rect)


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


def plot_histogram():
    if csv_type:
        target_pos, distance, episode_rew = read_csv(dir_csv)
    else:
        target_pos, distance, episode_rew = read_pickle_mean(dir_pickle)

    distance = np.array(distance)
    counts, bin_edges = np.histogram(distance, bins=np.arange(0, 800, 20))
    # counts, bin_edges = np.histogram(distance, bins=np.arange(0, 80, 5))
    # print(counts)
    plt.style.use('ggplot')
    x = len(counts)
    # histogram = plt.hist(distance, bins=np.arange(0, 800, 20), color='steelblue', align='mid', edgecolor='k', alpha=0.7)
    # histogram = plt.bar(np.arange(x)+0.45, counts, width=0.9, color='steelblue')
    histogram = plt.bar(np.arange(x)+0.5, counts, width=1, color='steelblue', edgecolor='k', linewidth=0.4, alpha=0.8)
    for hist, count in zip(histogram, counts):
        h01 = hist.get_height()
        plt.text(hist.get_x(), h01, count, fontsize=5.5, va='bottom')
        # plt.text(hist.get_x()+0.2, h01, count, fontsize=8, va='bottom')
    plt.xticks(np.arange(0, x+5, 5), np.arange(0, 900, 100), fontsize=10)
    # plt.xticks(np.arange(0, x+2, 2), np.arange(0, 90, 10), fontsize=10)
    plt.xlabel('Distance(mm)')
    plt.ylabel('Numbers(1000epi)')
    plt.title(exp_id)
    # file_name = './pictures/histogram/R6_standard/ppo/random' + r'/' + exp_id + '.png'
    # plt.savefig(file_name)
    plt.savefig(file_dir + 'plot/histogram.png')
    # plt.show()


def plot_boxes(exp_ids, labels, xlabel):
    distances, episode_rews = read_data(exp_ids)
    # mpl.rcParams['boxplot.flierprops.marker'] = '+'
    mpl.rcParams['boxplot.flierprops.markersize'] = 3
    mpl.rcParams['boxplot.meanprops.linestyle'] = '--'
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(9, 8))
    bplot1 = axes[0].boxplot(distances,
                             whis=[10, 90],
                             widths=0.8,
                             vert=True,  # vertical box alignment
                             showmeans=True,
                             patch_artist=True,  # fill with color
                             # showfliers=False,  # show outliers
                             labels=labels
                                )  # will be used to label x-ticks

    # axes[0].set_title('')
    axes[0].yaxis.grid(True)
    axes[0].set_xlabel(xlabel)
    axes[0].set_ylabel('Distance(mm)')

    bplot2 = axes[1].boxplot(episode_rews,
                             whis=[10, 90],
                             widths=0.8,
                             vert=True,  # vertical box alignment
                             showmeans=True,
                             patch_artist=True,  # fill with color
                             # showfliers=False,  # show outliers
                             labels=labels
                                )  # will be used to label x-ticks
    # axes[1].set_title('')
    axes[1].yaxis.grid(True)
    axes[1].set_xlabel(xlabel)
    axes[1].set_ylabel('episode_rew')

    # fill with colors
    colors = ['pink', 'lightblue', 'lightgreen']
    for bplot in (bplot1, bplot2):
        for box, color in zip(bplot['boxes'], colors):
            box.set(color='gray', linewidth=0.5)
            box.set(facecolor=color)
        for whisker, cap in zip(bplot['whiskers'], bplot['caps']):
            whisker.set(color='gray', linewidth=0.5, linestyle='--')
            cap.set(color='gray', linewidth=1)
    plt.subplots_adjust(bottom=0.15, wspace=0.3)
    fig.suptitle('Random init robot pose testing',x=0.45,y=0.95)
    # plt.show()


# 调整violin图中的whiskers
def adjacent_values(vals, q1, q3):
    upper_adjacent_value = q3 + (q3 - q1) * 1.5
    upper_adjacent_value = np.clip(upper_adjacent_value, q3, vals[-1])
    lower_adjacent_value = q1 - (q3 - q1) * 1.5
    lower_adjacent_value = np.clip(lower_adjacent_value, vals[0], q1)
    return lower_adjacent_value, upper_adjacent_value


def plot_violins(exp_ids, labels, xlabel):
    distances, episode_rews = read_data(exp_ids)
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(9, 8))
    vplot1 = axes[0].violinplot(distances,
                                widths=0.8,
                                showextrema=True,
                                # showmedians=True,
                                showmeans=True
                                )  # will be used to label x-ticks
    # axes[0].set_title('')
    axes[0].yaxis.grid(True)
    axes[0].set_xlabel(xlabel)
    axes[0].set_ylabel('Distance(mm)')

    vplot2 = axes[1].violinplot(episode_rews,
                                widths=0.8,
                                showextrema=True,
                                # showmedians=True,
                                showmeans=True
                                )  # will be used to label x-ticks
    # axes[1].set_title('')
    axes[1].yaxis.grid(True)
    axes[1].set_xlabel(xlabel)
    axes[1].set_ylabel('episode_rew')

    # # 设置violin图的颜色
    # for ax in axes:
    #     pc = ax['bodies']
    #     pc.set_facecolor('#D43F3A')
    #     pc.set_edgecolor('black')
    #     pc.set_alpha(1)

    data = [distances, episode_rews]
    for i in range(len(data)):
        quartile1, medians, quartile3 = np.percentile(data[i], [25, 50, 75], axis=0)
        # # 调整whiskers
        # whiskers = np.array([
        #     adjacent_values(sorted_array, q1, q3)
        #     for sorted_array, q1, q3 in zip(data[i], quartile1, quartile3)])
        # whiskersMin, whiskersMax = whiskers[:, 0], whiskers[:, 1]

        inds = np.arange(1, len(medians) + 1)
        axes[i].scatter(inds, medians, marker='o', color='white', s=10, zorder=3)
        axes[i].vlines(inds, quartile1, quartile3, color='lightgreen', linestyle='-', lw=5)
        # # 画出调整whiskers后的分位区间
        # axes[i].vlines(inds, whiskersMin, whiskersMax, color='pink', linestyle='-', lw=2)

    # set style for the axes
    plt.subplots_adjust(bottom=0.15, wspace=0.3)
    plt.setp(axes, xticks=[y + 1 for y in range(distances.shape[1])],
             xticklabels=labels)
    fig.suptitle('Random init robot pose testing',x=0.45,y=0.95)
    plt.show()



def plot_2d_scatter():
    if csv_type:
        target_pos, distance, episode_rew = read_csv(dir_csv)
    else:
        target_pos, distance, episode_rew = read_pickle_mean(dir_pickle)

    target_pos = np.array(target_pos)
    distance = np.array(distance)

    real_x = target_pos[:, 0]
    real_z = target_pos[:, 2]
    fig = plt.figure(figsize=(13,10))
    ax = fig.add_subplot(1, 1, 1)
    # scatter plot
    _cmap = 'gist_rainbow'
    print('*****',max(distance))
    # # 调整散点图点的大小设置，便于最大值相差较大的不同散点图之间的对比
    # # m=1200,s/20;m=200,s/3
    # m = 1200
    # if max(distance) > 1200:
    #     print('max(distance) > 1200')
    #     m = max(distance)
    # dis = ax.scatter(real_x, real_z, s=(m - np.array(distance)) / 20,
    #                  c=distance, cmap=_cmap, edgecolors='none', vmin=0.0, vmax=80)
    dis = ax.scatter(real_x, real_z, s=(max(distance)-np.array(distance))/10,
                c=distance, cmap=_cmap, edgecolors='none', vmin=0.0, vmax=80)
    fig.colorbar(dis, ax=ax)

    # contour_plot(real_x, real_z, distance, _cmap)
    # plot_circle()

    plt.axvline(max(real_x), linestyle='--', linewidth=2)
    plt.axvline(min(real_x), linestyle='--', linewidth=2)
    plt.axhline(max(real_z), linestyle='--', linewidth=2)
    plt.axhline(min(real_z), linestyle='--', linewidth=2)

    # plot_range_box()
    MaxDis = str(max(distance))
    fig.suptitle("Exp_id="+exp_id+"   "+"Mean Dis="+str(np.mean(distance))+ "   " + "Max Dis=" + MaxDis,x=0.45,y=0.95)
    plt.xlabel('x_axis[mm]')
    plt.ylabel('z_axis[mm]')
    # file_name = './pictures/scatter/R2_random' + r'/' + exp_id + '.png'
    # plt.savefig(file_name)
    plt.savefig(file_dir + 'plot/distribute.png')
    # plt.subplots_adjust(left=0.08, bottom=0.08, right=1.06, top=0.9)
    # plt.show()


def plot_3d_scatter():
    if csv_type:
        target_pose, distance, episode_rew = read_csv(dir_csv)
    else:
        target_pos, distance, episode_rew = read_pickle_mean(dir_pickle)
    target_pos = np.array(target_pos)
    distance = np.array(distance)

    real_x = target_pos[:, 0]
    real_y = target_pos[:, 1]
    real_z = target_pos[:, 2]

    fig = plt.figure(figsize=(13, 10))
    bx = [0, 0, 0, 0]
    bx[3] = fig.add_subplot(221, projection='3d')
    # scatter plot
    ddd = bx[3].scatter3D(real_x, real_y, real_z, s=(max(distance) - np.array(distance)) / 60, c=distance,
                          edgecolors='none', cmap='jet', vmin=0.0, vmax=300)
    # fig.colorbar(ddd, ax=ax)

    bx[3].set_xlabel('x_axis[mm]')
    bx[3].set_ylabel('y_axis[mm]')
    bx[3].set_zlabel('z_axis[mm]')

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

    for xyz_type in range(3):
        bx[xyz_type] = fig.add_subplot(sub_num[xyz_type])
        # scatter plot
        xyzd[xyz_type] = bx[xyz_type].scatter(real_xyz[xyz_type][0], real_xyz[xyz_type][1], s=(max(distance)-np.array(distance))/40,
                                              c=distance, cmap='jet', edgecolors='none', vmin=0.0, vmax=300)

        plt.axvline(max(real_xyz[xyz_type][0]), linestyle='--', linewidth=2)
        plt.axvline(min(real_xyz[xyz_type][0]), linestyle='--', linewidth=2)
        plt.axhline(max(real_xyz[xyz_type][1]), linestyle='--', linewidth=2)
        plt.axhline(min(real_xyz[xyz_type][1]), linestyle='--', linewidth=2)

        bx[xyz_type].set_xlabel(label_name[xyz_type][0])
        bx[xyz_type].set_ylabel(label_name[xyz_type][1])

    fig.suptitle("Exp_id="+exp_id+"   "+"Mean Dis="+str(np.mean(distance)),x=0.45,y=0.95)
    fig.colorbar(xyzd[xyz_type], ax=bx)
    # file_name = './pictures/scatter/R6_standard/ppo' + r'/' + exp_id + '.png'
    # plt.savefig(file_name)
    plt.savefig(file_dir + 'plot/distribute.png')
    # plt.subplots_adjust(left=0.08, bottom=0.08, right=1.06, top=0.9)
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
    plt.plot(cc_x, cc_z, 'k-', linewidth=2)

def plot_theta():
    _cmap = 'tab20b'
    dir_list = os.listdir(file_dir+'plot/')
    for cur_file in dir_list:
        if cur_file.endswith('obs_clip.txt'):
            name = cur_file.split(sep='o')
            pic_name = name[0]
            b = read_pickle_plot(pic_name, 'reward.txt')
            theta = np.array(b['theta'])
            reward = np.array(b['rew'])
            # print(np.shape(theta1))
            # print(np.shape(theta2))
            # print(np.shape(reward))
            T1 = theta[:, 0]
            T2 = theta[:, 1]
            print("Done:", pic_name)
            fig = plt.figure(figsize=(9,7))
            # fig = plt.figure(figsize=(20,5))
            ax = fig.add_subplot(1, 1, 1)
            rew = ax.scatter(T1, T2, c=reward, cmap=_cmap, edgecolors='none', vmin=-1,vmax=1)
            fig.colorbar(rew, ax=ax)
            currentAxis = plt.gca()
            rect = patches.Rectangle((-5, -5), 10, 10, linewidth=1, edgecolor='r', facecolor='none')
            currentAxis.add_patch(rect)
            # plot_action(pic_name)
            plot_obs(pic_name)
            plt.xlabel('theta1')
            plt.ylabel('theta2')
            # plt.xlim((-60, 135))
            # plt.ylim((-215, 45))
            dis = read_pickle_plot(pic_name, 'distance.txt')
            fig.suptitle("Target pos:"+pic_name+" // Distance:"+str(round(dis,2)),x=0.45,y=0.95)
            plt.savefig(file_dir+'plot/'+pic_name+'.png')
            # plt.subplots_adjust(left=0.13, bottom=0.13, right=1, top=0.88)
            # plt.show()

def plot_action(pic_name):

    a = read_pickle_plot(pic_name, 'action.txt')
    print(a[0])
    theta = np.zeros((np.shape(a)[0]+1, np.shape(a)[1]))
    for i in range(np.shape(a)[0]+1):
        if i > 0:
            theta[i][0] = theta[i-1][0] + a[i-1][0]
            theta[i][1] = theta[i-1][1] + a[i-1][1]

    plt.plot(theta[:, 0], theta[:, 1], 'k', linewidth=1)
    # plt.show()

def plot_obs(pic_name):

    obs_clip = read_pickle_plot(pic_name, 'obs_clip.txt')
    obs_clip = np.array(obs_clip)
    # print(obs_clip)
    l=plt.plot(obs_clip[:, 0], obs_clip[:, 1], 'ko')
    plt.setp(l, markerfacecolor='r')
    if osp.isfile(file_dir+'plot/'+pic_name+'obs_no_clip.txt'):
        obs_no_clip = read_pickle_plot(pic_name, 'obs_no_clip.txt')
        obs_no_clip = np.array(obs_no_clip)
        plt.plot(obs_no_clip[:, 0], obs_no_clip[:, 1], 'b.', linewidth=0.1)
    plt.show()

# 6轴机械手末端器（obs）3D运动轨迹图
def plot_obs3D():
    dir_list = os.listdir(file_dir+'plot/')
    for cur_file in dir_list:
        if cur_file.endswith('obs_clip.txt'):
            name = cur_file.split(sep='o')
            pic_name = name[0]
            target = pic_name.split('_')
            for i in range(len(target)):
                target[i] = int(target[i])
            fig = plt.figure(figsize=(9, 7))
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(target[0], target[1], target[2], color='r')
            obs_clip = read_pickle_plot(pic_name, 'obs_clip.txt')
            obs_clip = np.array(obs_clip)
            # print(obs_clip)
            l= ax.scatter3D(obs_clip[:, 6]+target[0], obs_clip[:, 7]+target[1], obs_clip[:, 8]+target[2], edgecolors='pink', facecolor='b', linewidth=0.5)
            # plt.setp(l, color='r')
            if osp.isfile(file_dir+'plot/'+pic_name+'obs_no_clip.txt'):
                obs_no_clip = read_pickle_plot(pic_name, 'obs_no_clip.txt')
                obs_no_clip = np.array(obs_no_clip)
                ax.scatter3D(obs_no_clip[:, 0], obs_no_clip[:, 1], obs_no_clip[:, 2], 'b.', linewidth=0.5)
            plt.subplots_adjust(left=0.13, bottom=0.13, right=1, top=0.88)
            ax.set_xlabel('x_axis[mm]')
            ax.set_ylabel('y_axis[mm]')
            ax.set_zlabel('z_axis[mm]')
            dis = read_pickle_plot(pic_name, 'distance.txt')
            fig.suptitle("Target pos:"+pic_name+" // Distance:"+str(round(dis,2)),x=0.45,y=0.95)
            plt.subplots_adjust(left=0.08, bottom=0.08, right=0.9, top=0.9)
            plt.savefig(file_dir+'plot/'+pic_name+'.png')
            # plt.show()

def plot_var():
    dir_list = os.listdir(file_dir + 'plot/')
    for cur_file in dir_list:
        if cur_file.endswith('var.txt'):
            name = cur_file.split(sep='v')
            pic_name = name[0]
            varinfo = read_pickle_plot(pic_name, 'var.txt')
            var = varinfo["var"]
            act = varinfo["action"]
            var = np.array(var)
            act = np.array(act)

            # print(np.shape(var[:,0,0]))
            # print(np.shape(act[0]))
            fig = plt.figure(figsize=(10,10))
            ax = fig.add_subplot(2,2,1)
            F_var, =ax.plot(act[:,0, 0], 'r.', label='gaussian_fixed_var:False')
            ax.set_title("axis_2")
            ax.set_xlabel("step")
            ax.set_ylabel("action")

            bx = fig.add_subplot(2,2,2)
            bx.plot(act[:,0, 1], 'r.', linewidth=0.1)
            bx.set_title("axis_3")
            bx.set_xlabel("step")
            bx.set_ylabel("action")
            # fig2 = plt.figure(figsize=(10,5))
            cx = fig.add_subplot(2,2,3)
            cx.plot(var[:,0, 0], 'r.')
            cx.set_xlabel("step")
            cx.set_ylabel("var")
            dx = fig.add_subplot(2,2,4)
            dx.plot(var[:,0, 1], 'r.')
            dx.set_xlabel("step")
            dx.set_ylabel("var")
            fig.suptitle("Target pos:"+pic_name)
            if osp.isfile(file_dir+'plot/'+pic_name+'var2.txt'):
                varinfo2 = read_pickle_plot(pic_name, 'var2.txt')
                var2 = varinfo2["var"]
                act2 = varinfo2["action"]
                var2 = np.array(var2)
                act2 = np.array(act2)
                T_var, =ax.plot(act2[:, 0, 0], 'b.', label='gaussian_fixed_var:True')
                bx.plot(act2[:, 0, 1], 'b.', linewidth=0.1)
                cx.plot(var2[:, 0, 0], 'b.')
                dx.plot(var2[:, 0, 1], 'b.')
                fig.legend(handles=[F_var, T_var])
                plt.savefig(file_dir + 'plot/' + pic_name + '_var_compare.png')
            else:

                plt.savefig(file_dir+'plot/'+pic_name+'_var.png')
            plt.show()


if __name__ == '__main__':
    exp_ids = [['20190108-0452', '20190102-1832', '20190108-2137',
                '20190107-2119', '20190103-0052', '20190108-1751',
                '20190108-1127', '20190103-2324', '20190109-1208',
                '20190108-0105', '20190104-0549', '20190109-0737'],
               ['20190110-0414', '20190111-2239', '20190112-0403',
                '20190109-2226', '20190111-1947', '20190112-0110',
                '20190110-0647', '20190112-0815', '20190112-1523',
                '20190110-0120', '20190112-0635', '20190112-1127'],
               ['20190107-1304', '20190104-1350', '20190105-2058',
                '20190107-1013', '20190104-1830', '20190105-2311',
                '20190102-0819', '20190105-0956', '20190106-0611',
                '20190107-1522', '20190105-1507', '20190106-0823']]
    xlabels = ['128', '256', '512']
    xlabel = 'optim_batchsize'
    # plot_violins(exp_ids, xlabels, xlabel)
    # plot_boxes1(exp_ids, xlabels, xlabel)
    # plot_boxes(exp_ids, xlabels, xlabel)

    plot_histogram()
    # plot_2d_scatter()
    # plot_3d_scatter()

    # plot_theta()
    # plot_obs3D()

    # dis_xy = np.zeros([10, 3])
    # dis_xy = [[0 for i in range(10)] for j in range(3)]
    # a = [1,1,1,1,1,1,1,1,1,1]
    # x = dis_xy[2]
    # dis_xy[1]=a
    # dis_xy = np.array(dis_xy)
    # print(dis_xy[1,:])
    # plot_reward()


    # test()
    # read_pickle()
    # plot_action()
    # plot_obs()
    # plot_var()
