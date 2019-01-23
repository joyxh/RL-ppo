# -*- coding: utf-8 -*-
from gym import logger
from gym import utils, spaces
import gym
from gym.utils import seeding

import numpy as np
import random
import yaml
import math
import copy
import sys
sys.path.append('./rllib/')
from envs.utils.PyKinematics.SixKin import SixAxisKin

ROBOT_CONFIG_PATH = '/root/gzb/robot_learning/config/robot_config.yaml'
ENV_CONFIG_PATH = 'config/env_config.yaml'

JOINT_ANGLE_LIMIT = 5
all_pos = [[-100, 0, 800],
           [-200, 0, 700],
           [-300, 0, 400],
           [80, 0, 900],
           [150, 0, 0],
           [200, 0, 650],
           [380, 0, 380],
           [200, 0, 200],
           [400, 0, 500],
           [650, 0, 380],
           [-100, 0, 100]]
# given_obj_pos = all_pos[10]  # 0-10 0.3.5.7
clip_j = True       # whether to clip joint angle
new_clip = False    # joint angle +/- 360 when out of range


class ReacherBenchmarkEnv(gym.Env):
    def __init__(self):
        # load config
        with open(ROBOT_CONFIG_PATH, 'r') as robot_config_file:
            self.robot_config = yaml.load(robot_config_file)
        with open(ENV_CONFIG_PATH, 'r') as env_config_file:
            self.env_config = yaml.load(env_config_file)

        # setup fk, including robot joint angle limit
        self.robot_name = self.env_config['robot_name']
        self._set_fk_config()

        # setup target generation
        assert self.env_config['random_target_method'] in ['gaussian', 'uniform']
        assert self.env_config['random_target_mode'] in ['pose', 'joint_angle']

        self.random_target_method = self.env_config['random_target_method']
        self.random_target_mode = self.env_config['random_target_mode']

        self.X_LIMIT = self.env_config['X_LIMIT']
        self.Y_LIMIT = self.env_config['Y_LIMIT']
        self.Z_LIMIT = self.env_config['Z_LIMIT']

        self._random_x_method = self._get_random_x_method(self.env_config['random_target_method'])
        self.given_obj_pos = None

        # setup robot init pose
        # self._init_robot_pos_rad = [sum(X_LIMIT) / 2, sum(Y_LIMIT) / 2, sum(Z_LIMIT) / 2, 0, -1.5708, -3.1416]
        # self._init_robot_ja_6x1 = self._get_init_robot_ja(self._init_robot_pos_rad)
        # self._init_robot_ja_6x1 = [0, 0, 0, 0, 0, 0]
        self.given_init_robot_ja_6x1 = self.env_config['given_init_robot_ja_6x1']
        self._init_robot_ja_6x1 = [0] * 6

        # other params
        self.dof_ctrl_index = self.env_config['dof_ctrl_index']
        self.NewRewF = self.env_config['NewRewF']

        # for gym
        self.action_space = spaces.Box(-JOINT_ANGLE_LIMIT, JOINT_ANGLE_LIMIT, shape=(len(self.dof_ctrl_index),),
                                       dtype='float32')
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(len(self.dof_ctrl_index) + 3,), dtype='float32')
        self.reward_range = (-np.inf, np.inf)

        # self.seed()

    def render(self, mode='human'):
        raise Exception('Graphics not available!')

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        # action is delta J
        action = self._clip_action(action)
        cur_joint_angle = self._set_joint_angle(action)
        self.end_effector_pos = self._get_end_effector_pos(self._all_joint_angle)
        logger.debug('Current Joint Angle: %r' % cur_joint_angle)
        logger.debug('Current End Effector Pos: %r' % self.end_effector_pos)

        # get obs, and update gripper pos.
        observation = self._get_obs()

        reward = self._get_reward(observation)

        info = {'object_pos': self.object_pos,
                'end_effector_pos': self.end_effector_pos,
                'joint_angle_pose': self._all_joint_angle
                }

        done = False
        return observation, reward, done, info

    def reset(self):
        """
        Resets the robot joint angle to [0,0,0,0,0,0]
        And Randomly initialize the position of the object
        """
        # At the same time, update the object pos to self.object_pos.
        self._reset_obj_pos()
        self._reset_robot_pos()
        self._all_joint_angle = copy.deepcopy(self._init_robot_ja_6x1)
        self.end_effector_pos = copy.deepcopy(self._init_robot_pos_rad)
        logger.info("Reset Robot Joint Angle : %r ." % self._all_joint_angle)
        logger.debug("Reset Robot end_effector_pos : %r ." % self.end_effector_pos)

        # return obs !
        return self._get_obs()

    def is_given_obj_pos_in_range(self):
        assert (len(self.given_obj_pos) == 3)  # x, y, z
        assert (self.X_LIMIT[0] <= self.given_obj_pos[0] <= self.X_LIMIT[1])
        assert (self.Y_LIMIT[0] <= self.given_obj_pos[1] <= self.Y_LIMIT[1])
        assert (self.Z_LIMIT[0] <= self.given_obj_pos[2] <= self.Z_LIMIT[1])

    def _get_init_robot_ja(self, init_robot_pos_rad):
        ref_joints_rad = np.zeros(6)
        result_joints_rad = np.zeros(6)
        weight = self.robot_config[self.robot_name]['Axis6']['weight']

        ref_value = self.kin.ikSolution(init_robot_pos_rad, weight, ref_joints_rad, result_joints_rad, 2)

        if ref_value < 0:
            logger.ERROR('Init Robot Pose Error !!!!!!!!!!')

        init_robot_ja_6x1 = np.rad2deg(result_joints_rad).tolist()
        return init_robot_ja_6x1

    def _clip_action(self, action):
        assert (len(action) == len(self.dof_ctrl_index))
        clip_action = np.clip(action, -JOINT_ANGLE_LIMIT, JOINT_ANGLE_LIMIT)
        return clip_action

    def _get_reward(self, obs):
        dx_m = obs[-3] * 0.001
        dy_m = obs[-2] * 0.001
        dz_m = obs[-1] * 0.001
        distance_m = math.sqrt(dx_m ** 2 + dy_m ** 2 + dz_m ** 2)
        if not self.NewRewF:
            reward = -distance_m + math.exp(-100 * distance_m * distance_m)
        else:
            reward = -distance_m + math.exp(-10 * distance_m)

        return reward

    def _reset_obj_pos(self):
        if self.given_obj_pos:
            self.is_given_obj_pos_in_range()
            self.object_pos = self.given_obj_pos
        elif 'joint_angle' == self.random_target_mode:
            max_jnt = self.robot_config[self.robot_name]['Axis6']['jointUpperLimit']
            min_jnt = self.robot_config[self.robot_name]['Axis6']['jointLowerLimit']
            _all_ja_for_reset = copy.deepcopy(self._init_robot_ja_6x1)
            for i in range(len(self.dof_ctrl_index)):
                ja_limit = (min_jnt[self.dof_ctrl_index[i]], max_jnt[self.dof_ctrl_index[i]])
                _all_ja_for_reset[self.dof_ctrl_index[i]] = self._random_x_method(ja_limit)
            reset_pos_rad = self._get_end_effector_pos(_all_ja_for_reset)
            self.object_pos = reset_pos_rad[:3]

        else:  # 'pose' == self.random_target_mode:
            # X_LIMIT: experience value
            x_mm = self._random_x_method(self.X_LIMIT)
            y_mm = self._random_x_method(self.Y_LIMIT)
            z_mm = self._random_x_method(self.Z_LIMIT)

            # 2 axis : xz
            if 2 == len(self.dof_ctrl_index):
                # y_mm = sum(self.Y_LIMIT) / 2
                y_mm = 0

            self.object_pos = [x_mm, y_mm, z_mm]
        logger.info("*****Reset Object Pos: %r ." % self.object_pos)
        # print("***Reset Object Pos:", _all_ja_for_reset)

    def _reset_robot_pos(self):
        if self.given_init_robot_ja_6x1:
            self._init_robot_ja_6x1 = self.given_init_robot_ja_6x1
        else:
            max_jnt = self.robot_config[self.robot_name]['Axis6']['jointUpperLimit']
            min_jnt = self.robot_config[self.robot_name]['Axis6']['jointLowerLimit']
            _all_ja_for_reset = copy.deepcopy(self._init_robot_ja_6x1)
            for i in range(len(self.dof_ctrl_index)):
                ja_limit = (min_jnt[self.dof_ctrl_index[i]], max_jnt[self.dof_ctrl_index[i]])
                _all_ja_for_reset[self.dof_ctrl_index[i]] = self._random_x_method(ja_limit)
            self._init_robot_ja_6x1 = _all_ja_for_reset

        self._init_robot_pos_rad = self._get_end_effector_pos(self._init_robot_ja_6x1)
        logger.info("*****Reset init robot ja: %r ." % self._init_robot_ja_6x1)

    def _set_fk_config(self):
        max_jnt = self.robot_config[self.robot_name]['Axis6']['jointUpperLimit']
        min_jnt = self.robot_config[self.robot_name]['Axis6']['jointLowerLimit']
        rot_dir = self.robot_config[self.robot_name]['Axis6']['rotDir']
        a = self.robot_config[self.robot_name]['Axis6']['dH_a']
        d = self.robot_config[self.robot_name]['Axis6']['dH_d']

        self.kin = SixAxisKin(a, d, rot_dir)
        self.kin.setLimit(min_jnt, max_jnt)

    def _set_joint_angle(self, new_ja):
        """
        set and clip joint angle accoring to zhe JointUpperLimit & JointLowerLimit.
        :param new_ja: action
        :return:
        """
        max_jnt = self.robot_config[self.robot_name]['Axis6']['jointUpperLimit']
        min_jnt = self.robot_config[self.robot_name]['Axis6']['jointLowerLimit']

        # for debug:
        _all_ja_for_judge = copy.deepcopy(self._all_joint_angle)
        # print(_all_ja_for_judge)

        for i in range(len(self.dof_ctrl_index)):
            _all_ja_for_judge[self.dof_ctrl_index[i]] += new_ja[i]
        # print(_all_ja_for_judge)
        ja_max_than_max_flag = np.array(_all_ja_for_judge) > np.array(max_jnt)
        ja_less_than_min_flag = np.array(_all_ja_for_judge) < np.array(min_jnt)
        if ja_less_than_min_flag.any() or ja_max_than_max_flag.any():
            logger.debug('Joint Angle Out Of Limit : %r' % _all_ja_for_judge)

        if clip_j:
            # clip ja:
            for index in range(len(self.dof_ctrl_index)):
                if new_clip:
                    next_ja = self._all_joint_angle[self.dof_ctrl_index[index]] + new_ja[index]
                    if next_ja > max_jnt[self.dof_ctrl_index[index]]:
                        cur_ja = next_ja - 360
                    elif next_ja < min_jnt[self.dof_ctrl_index[index]]:
                        cur_ja = next_ja + 360
                    else:
                        cur_ja = next_ja
                else:
                    cur_ja = np.clip(self._all_joint_angle[self.dof_ctrl_index[index]] + new_ja[index],
                                     min_jnt[self.dof_ctrl_index[index]], max_jnt[self.dof_ctrl_index[index]])

                self._all_joint_angle[self.dof_ctrl_index[index]] = cur_ja
            current_joint_angle = copy.deepcopy(self._all_joint_angle)[self.dof_ctrl_index[0]:self.dof_ctrl_index[-1] + 1]
        else:
            for index in range(len(self.dof_ctrl_index)):
                self._all_joint_angle[self.dof_ctrl_index[index]] = _all_ja_for_judge[self.dof_ctrl_index[index]]
            current_joint_angle = copy.deepcopy(_all_ja_for_judge)[self.dof_ctrl_index[0]:self.dof_ctrl_index[-1] + 1]
            # print(new_ja)

        return current_joint_angle

    def _get_end_effector_pos(self, joint_angle_6x1):
        input_jnt = np.array(joint_angle_6x1, np.float)
        input_jnt_rad = np.deg2rad(input_jnt)
        out_coord_rad = np.zeros(6)
        self.kin.forwardKin(input_jnt_rad, out_coord_rad)
        # attention: u v w is rad;
        return out_coord_rad

    def _get_obs(self):
        """Returns the observation.
        """
        distance_x = self.end_effector_pos[0] - self.object_pos[0]
        distance_y = self.end_effector_pos[1] - self.object_pos[1]
        distance_z = self.end_effector_pos[2] - self.object_pos[2]
        obs = copy.deepcopy(self._all_joint_angle)[self.dof_ctrl_index[0]:self.dof_ctrl_index[-1] + 1]
        obs.extend([distance_x, distance_y, distance_z])
        return np.array(obs)

    @staticmethod
    def _get_random_x_method(method):
        def gaussian_x(x_limit):
            """
            generate random x coordinates using Gaussian distribution;
            the mean of the Gaussian if set to the middle of the max - min range;
            the std of the Gaussian is set to a quarter of the max - min range;
            rejection sampling is used if the generated coordinates exceed either limit
            :param x_limit: a tuple (x_min, x_max) indicating the range in x-axis
            :return: x
            """
            x_min, x_max = x_limit
            assert x_min <= x_max, 'x_min <= x_max (%.4f <= %.4f)' % (x_min, x_max)
            x = np.random.randn() * (x_max - x_min) / 4 + (x_max + x_min) / 2
            if not x_min <= x <= x_max:
                x = gaussian_x(x_limit)
            return x

        def uniform_x(x_limit):
            """
            generate random x coordinates using Uniform distribution
            :param x_limit: a tuple (x_min, x_max) indicating the range in x-axis
            :return: x
            """
            x_min, x_max = x_limit
            x = np.random.rand() * (x_max - x_min) + x_min
            return x

        if method == 'gaussian':
            return gaussian_x
        elif method == 'uniform':
            return uniform_x
        else:
            return None

    def test_robot_pos(self):
        test_pos = [450, 0, 350, 0, 0, 0]
        ref_joints_rad = np.zeros(6)
        result_joints_rad = np.zeros(6)

        weight = self.robot_config[self.robot_name]['Axis6']['weight']
        ref_value = self.kin.ikSolution(test_pos, weight, ref_joints_rad, result_joints_rad, 2)

        if ref_value < 0:
            print('无解')
        else:
            print('ik solution flags:', ref_value)
            print('ik result joint angle:', np.rad2deg(result_joints_rad))
            out_coord = np.zeros(6)
            self.kin.forwardKin(result_joints_rad, out_coord)
            print('ik end effector pos:', out_coord)
            print('test end effector pos:', test_pos)


if __name__ == '__main__':
    # logger.set_level(logger.DEBUG)

    dof_ctrl_index = [1, 2]
    # dof_ctrl_index = [0, 1, 2, 3, 4, 5]
    robot_name = 'PIROBOT'
    env = ReacherBenchmarkEnv()

    # env.get_reset_obj_pos(given_obj_pos=[600, 1, 200])
    env.reset()
    for i in range(10):
        # action = env.action_space.sample()
        action = [1, 10]
        # action = [1,10,3,0,0,0]
        observation, reward, done, info = env.step(action)
        print(reward)
        # print(observation)
        # print(info)

    # env.test_robot_pos()

    # after `pip install -e .`:
    # env = gym.make('Reacher2BenchmarkEnv-v0')
    # env.reset()