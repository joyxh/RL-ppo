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
import time

from envs.gazebo.gazebo_env import GazeboEnv
from envs.gazebo.GazeboInterface.RobotControl import RobotControl
from envs.gazebo.GazeboInterface.WorldControl import WorldControl
from envs.utils.PyKinematics.SixKin import SixAxisKin

ROBOT_CONFIG_PATH = 'config/robot_config.yaml'
ENV_CONFIG_PATH = 'config/reacher_gazebo_env_config.yaml'
ROBOT_NAME = 'rr_arm'
DOF = 6
ROBOT_BASE_LINK = 'rr_arm::rr6::table_link'
TARGET_LINK = 'ball::link'
TARGET_MODEL = 'ball'

JOINT_ANGLE_LIMIT = 5
# INIT_ROBOT_POS_DEGREE = [sum(X_LIMIT) / 2, sum(Y_LIMIT) / 2, sum(Z_LIMIT) / 2, 0, -90, -180]
INIT_ROBOT_JOINT_ANGLE = [0, 0, 0, 0, 0, 0]
JOINT_LIMIT_DELTA = 0.01


class ReacherBenchmarkGazeboEnv(GazeboEnv):
    def __init__(self, robot_name='PIROBOT'):
        with open(ROBOT_CONFIG_PATH, 'r') as robot_config_file:
            self.robot_config = yaml.load(robot_config_file)
        with open(ENV_CONFIG_PATH, 'r') as env_config_file:
            self.env_config = yaml.load(env_config_file)
        self.robot_name = robot_name
        self._set_fk_config()

        # Launch the simulation with the given launchfile name
        launchfile_path = 'reacher_env.launch'
        GazeboEnv.__init__(self, launchfile_path)

        self.dof_ctrl_index = self.env_config['dof_ctrl_index']
        self.NewRewF = self.env_config['NewRewF']


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
        self.given_init_robot_ja_6x1 = self.env_config['given_init_robot_ja_6x1']
        # setup robot
        self.robot = RobotControl(robot_name=ROBOT_NAME, dof=DOF)
        self._world = WorldControl()
        self._robot_base_pos = self._world.getLinkPos(linkName=ROBOT_BASE_LINK)

        self.action_space = spaces.Box(-JOINT_ANGLE_LIMIT, JOINT_ANGLE_LIMIT, shape=(len(self.dof_ctrl_index),),
                                       dtype='float32')
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(len(self.dof_ctrl_index) + 3,), dtype='float32')
        self.reward_range = (-np.inf, np.inf)

        # self.seed()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        # action is delta J
        action = self._clip_action(action)
        cur_joint_angle = self._set_joint_angle(action)
        self.end_effector_pos = self._get_end_effector_pos()
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
        self.object_pos = self._world.getLinkPos(linkName=TARGET_LINK)
        self.object_pos[2] -= self._robot_base_pos[2]
        logger.info("Reset Object Pos: %r ." % self.object_pos)
        self._all_joint_angle = self.robot.getJointPos()
        self.end_effector_pos = self._get_end_effector_pos()

        logger.info("Reset Robot Joint Angle : %r ." % self._all_joint_angle)
        logger.debug("Reset Robot end_effector_pos : %r ." % self.end_effector_pos)

        # return obs !
        return self._get_obs()

    def is_given_obj_pos_in_range(self):
        assert (len(self.given_obj_pos) == 3)  # x, y, z
        assert (self.X_LIMIT[0] <= self.given_obj_pos[0] <= self.X_LIMIT[1])
        assert (self.Y_LIMIT[0] <= self.given_obj_pos[1] <= self.Y_LIMIT[1])
        assert (self.Z_LIMIT[0] <= self.given_obj_pos[2] <= self.Z_LIMIT[1])

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
            obj_pos_in_gazebo = copy.deepcopy(self.given_obj_pos)
            obj_pos_in_gazebo.extend([0, 0, 0])
            # z add robot_base
            obj_pos_in_gazebo[2] += self._robot_base_pos[2]
        elif 'joint_angle' == self.random_target_mode:
            max_jnt = self.robot_config[self.robot_name]['Axis6']['jointUpperLimit']
            min_jnt = self.robot_config[self.robot_name]['Axis6']['jointLowerLimit']
            _all_ja_for_reset = copy.deepcopy(INIT_ROBOT_JOINT_ANGLE)
            for i in range(len(self.dof_ctrl_index)):
                ja_limit = (min_jnt[self.dof_ctrl_index[i]] + JOINT_LIMIT_DELTA,
                            max_jnt[self.dof_ctrl_index[i]] - JOINT_LIMIT_DELTA)
                _all_ja_for_reset[self.dof_ctrl_index[i]] = self._random_x_method(ja_limit)
            # print(_all_ja_for_reset)
            reset_pos_rad = self._forward_kin(_all_ja_for_reset)
            obj_pos_in_gazebo = reset_pos_rad[:3].tolist()
            # print(obj_pos_in_gazebo)
            obj_pos_in_gazebo.extend([0, 0, 0])
            # z add robot_base
            obj_pos_in_gazebo[2] += self._robot_base_pos[2]
        else:  # 'pose' == self.random_target_mode:
            # X_LIMIT: experience value
            x_mm = self._random_x_method(self.X_LIMIT)
            y_mm = self._random_x_method(self.Y_LIMIT)
            z_mm = self._random_x_method(self.Z_LIMIT)

            # 2 axis : xz
            if 2 == len(self.dof_ctrl_index):
                # y_mm = sum(self.Y_LIMIT) / 2
                y_mm = 0

            obj_pos_in_gazebo = [x_mm, y_mm, z_mm + self._robot_base_pos[2], 0, 0, 0]

        self._world.setModelStatic(modelName=TARGET_MODEL, isStatic=True)
        reset_obj_success = self._world.setLinkPos(linkName=TARGET_LINK, pos=obj_pos_in_gazebo)
        if not reset_obj_success:
            self._reset_obj_pos()

    def _reset_robot_pos(self):
        if self.given_init_robot_ja_6x1:
            reset_robot_success = self.robot.setJointPos(self.given_init_robot_ja_6x1, wait=True)
        else:
            max_jnt = self.robot_config[self.robot_name]['Axis6']['jointUpperLimit']
            min_jnt = self.robot_config[self.robot_name]['Axis6']['jointLowerLimit']
            _all_ja_for_reset = copy.deepcopy(INIT_ROBOT_JOINT_ANGLE)
            for i in range(len(self.dof_ctrl_index)):
                ja_limit = (min_jnt[self.dof_ctrl_index[i]] + JOINT_LIMIT_DELTA,
                            max_jnt[self.dof_ctrl_index[i]] - JOINT_LIMIT_DELTA)
                _all_ja_for_reset[self.dof_ctrl_index[i]] = self._random_x_method(ja_limit)
            reset_robot_success = self.robot.setJointPos(_all_ja_for_reset, wait=True)
        if not reset_robot_success:
            self._reset_robot_pos()

    def _set_fk_config(self):
        max_jnt = self.robot_config[self.robot_name]['Axis6']['jointUpperLimit']
        min_jnt = self.robot_config[self.robot_name]['Axis6']['jointLowerLimit']
        rot_dir = self.robot_config[self.robot_name]['Axis6']['rotDir']
        a = self.robot_config[self.robot_name]['Axis6']['dH_a']
        d = self.robot_config[self.robot_name]['Axis6']['dH_d']

        self.kin = SixAxisKin(a, d, rot_dir)
        self.kin.setLimit(min_jnt, max_jnt)

    def _forward_kin(self, joint_angle_6x1):
        input_jnt = np.array(joint_angle_6x1, np.float)
        input_jnt_rad = np.deg2rad(input_jnt)
        out_coord_rad = np.zeros(6)
        self.kin.forwardKin(input_jnt_rad, out_coord_rad)
        # attention: u v w is rad;
        return out_coord_rad

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
        for i in range(len(self.dof_ctrl_index)):
            _all_ja_for_judge[self.dof_ctrl_index[i]] += new_ja[i]
        ja_max_than_max_flag = np.array(_all_ja_for_judge) > np.array(max_jnt)
        ja_less_than_min_flag = np.array(_all_ja_for_judge) < np.array(min_jnt)
        if ja_less_than_min_flag.any() or ja_max_than_max_flag.any():
            logger.debug('Joint Angle Out Of Limit : %r' % _all_ja_for_judge)

        # clip ja:
        for index in range(len(self.dof_ctrl_index)):
            cur_ja = np.clip(self._all_joint_angle[self.dof_ctrl_index[index]] + new_ja[index],
                             min_jnt[self.dof_ctrl_index[index]] + JOINT_LIMIT_DELTA,
                             max_jnt[self.dof_ctrl_index[index]] - JOINT_LIMIT_DELTA)
            self._all_joint_angle[self.dof_ctrl_index[index]] = cur_ja
        # set ja:
        self.robot.setJointPos(self._all_joint_angle, wait=True)
        time.sleep(0.1)
        self._all_joint_angle = self.robot.getJointPos()
        current_joint_angle = copy.deepcopy(self._all_joint_angle)[self.dof_ctrl_index[0]:self.dof_ctrl_index[-1] + 1]
        return current_joint_angle

    def _get_end_effector_pos(self):
        robot_pos = self.robot.getRobotPos()
        return robot_pos

    def _get_obs(self):
        """Returns the observation.
        """
        distance_x = self.end_effector_pos[0] - self.object_pos[0]
        distance_y = self.end_effector_pos[1] - self.object_pos[1]
        distance_z = self.end_effector_pos[2] - self.object_pos[2]
        obs = copy.deepcopy(self._all_joint_angle)[self.dof_ctrl_index[0]:self.dof_ctrl_index[-1] + 1].tolist()
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


if __name__ == '__main__':
    logger.set_level(logger.DEBUG)

    dof_ctrl_index = [1, 2]
    # dof_ctrl_index = [0, 1, 2, 3, 4, 5]
    robot_name = 'PIROBOT'
    env = ReacherBenchmarkGazeboEnv(random_target_mode='joint_angle', random_target_method='gaussian',
                                    dof_ctrl_index=dof_ctrl_index, robot_name=robot_name)

    # env.get_reset_obj_pos(given_obj_pos=[600, 1, 200])
    env.reset()
    for i in range(3):
        # action = env.action_space.sample()
        action = [1, 10]
        # action = [1,10,3,0,0,0]
        observation, reward, done, info = env.step(action)
        print(reward)
        print(observation)
        print(info)
        # after `pip install -e .`:
        # env = gym.make('Reacher2BenchmarkEnv-v0')
        # env.reset()
