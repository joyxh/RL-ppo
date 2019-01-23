import gym
import numpy as np
import time
import json
import envs
import pickle
from matplotlib import pyplot as plt
import sys
sys.path.append('/root/pick/robot_learning/')
from data_analyze.data_plot import file_dir, pic_name


def action_test():
    action = [0, 0]
    T1 = [-51,-200]


    env = gym.make('Reacher2BenchmarkGazeboEnv-v1')
    obs, done = env.reset(), False
    print(obs)
    action = T1
    obs, rew, done, info = env.step(action)
    # print(obs)

    # for i in range(23):
    #     action[1] -= 5
    #     obs, rew, done, info = env.step(action)
    #     print(obs)
    #     time.sleep(1)

def plot_circle():
    # 画边界线
    with open('experiments/circle.txt', 'rb') as file:
        b = pickle.load(file)
    b = np.array(b)
    cc_x = b[:, 0]
    cc_z = b[:, 2]
    plt.plot(cc_x, cc_z, 'r-')
    plt.show()

def plot_theta():
    action = [0, 0]
    # print(type(action))

    env = gym.make('Reacher2BenchmarkEnv-v1')
    obs, done = env.reset(), False
    theta1 = np.arange(-51, 126, 2)
    theta2 = np.arange(-206, 34, 2)
    # theta1 = np.arange(-180, 180, 2)
    # theta2 = np.arange(-270, 90, 2)
    # np.meshgrid生成网格点坐标矩阵
    T1 = np.array(np.meshgrid(theta1,theta2)).T.reshape(-1, 2)
    reward = np.zeros(np.shape(T1)[0])
    # print(list(T1[0]))

    for i in range(np.shape(T1)[0]):
        obs = env.reset()
        # action[0] = list(T1[i])[0]-obs[0]
        # action[1] = list(T1[i])[1]-obs[1]
        action = list(T1[i])

        obs, reward[i], done, info = env.step(action)
    # for i in range(len(T1)):
    #     action[0] = T1[i]
    #     action[1] = T2[i]
    #     print(action)
    #     obs, reward[i], done, info = env.step(action)

    infopic = {'theta':T1, 'rew':reward}
    with open('./' + file_dir+'plot/'+pic_name+'reward.txt', 'wb') as file:
        pickle.dump(infopic, file)




if __name__ == '__main__':
    # plot_circle()
    plot_theta()
    # action_test()
