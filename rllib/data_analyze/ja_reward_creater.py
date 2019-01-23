import gym
import numpy as np
import pickle
import os
from matplotlib import pyplot as plt
import sys
from data_plot import file_dir
sys.path.append("./rllib/")
from envs import wrappers
from envs.reacher_benchmark_env.reacher_benchmark import JOINT_ANGLE_LIMIT



def creat_theta():
    assert (JOINT_ANGLE_LIMIT == 500), "JOINT_ANGLE_LIMIT in reacher_benchmark.py must be 500!"
    env = gym.make('ReacherBenchmarkEnv-v1')
    env = wrappers.wrap_reacher(env)
    obs, done = env.reset(), False
    # theta1 = np.arange(-51, 126, 2)
    # theta2 = np.arange(-206, 34, 2)
    theta1 = np.arange(-180, 180, 2)
    theta2 = np.arange(-270, 90, 2)
    T1 = np.array(np.meshgrid(theta1,theta2)).T.reshape(-1, 2)
    reward = np.zeros(np.shape(T1)[0])
    # print(list(T1[0]))
    dir_list = os.listdir(file_dir+'plot/')
    for cur_file in dir_list:
        if cur_file.endswith('obs_clip.txt'):
            name = cur_file.split(sep='o')
            pic_name = name[0]
            given_pos_name = pic_name.split(sep='_')
            given_pos = [0] * 3
            for num in range(len(given_pos)):
                given_pos[num] = int(given_pos_name[num])
                for i in range(np.shape(T1)[0]):
                    obs = env.reset(given_pos)
                    action = list(T1[i])
                    obs, reward[i], done, info = env.step(action)

            infopic = {'theta':T1, 'rew':reward}
            with open(file_dir+'plot/'+pic_name+'reward.txt', 'wb') as file:
                pickle.dump(infopic, file)
            print("Finished:", given_pos)




if __name__ == '__main__':
    # plot_circle()
    creat_theta()
    # action_test()