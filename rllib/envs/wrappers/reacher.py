import gym
import copy
import numpy as np


class GiveObjPos(gym.Wrapper):
    """
    Reset object pos to definite value;

    """
    def __init__(self, env):
        gym.Wrapper.__init__(self, env)

    def step(self, action):

        return self.env.step(action)

    def reset(self, given_obj_pos=None):
        self.unwrapped.given_obj_pos = given_obj_pos
        obs = self.env.reset()
        return obs


def wrap_reacher(env):
    env = GiveObjPos(env)

    return env
