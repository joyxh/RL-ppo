import cv2
import gym
import copy
import numpy as np


class WarpAndStackImages(gym.ObservationWrapper):
    """Warp images to 84x84 and stack two images."""
    def __init__(self, env):
        gym.ObservationWrapper.__init__(self, env)
        self.width = 84
        self.height = 84
        self.__update_observation_space(env)

    def __update_observation_space(self, env):
        space_dict = copy.copy(env.observation_space.spaces)
        del space_dict['img_left1'], space_dict['img_left2']  # delete spaces of images
        space_dict['image'] = gym.spaces.Box(
            low=0, high=255, shape=(self.width, self.height, 3*2), dtype=np.uint8)  # add resized & stacked image space
        self.observation_space = gym.spaces.Dict(space_dict)

    def observation(self, obs):
        # get images and delete them from obs
        img1 = obs.pop('img_left1')
        img2 = obs.pop('img_left2')
        # resize images
        img1_resized = cv2.resize(img1, (self.width, self.height), interpolation=cv2.INTER_AREA)
        img2_resized = cv2.resize(img2, (self.width, self.height), interpolation=cv2.INTER_AREA)

        # stack images
        img_stacked = np.concatenate((img1_resized, img2_resized), axis=2)

        obs['image'] = img_stacked
        return obs


class RemoveRobotPose(gym.ObservationWrapper):
    def __init__(self, env):
        gym.ObservationWrapper.__init__(self, env)
        self.__update_observation_space(env)

    def __update_observation_space(self, env):
        space_dict = copy.copy(env.observation_space.spaces)
        self.observation_space = space_dict['image']

    def observation(self, obs):
        return obs['image']


class NoImageObservation(gym.ObservationWrapper):
    def __init__(self, env):
        gym.ObservationWrapper.__init__(self, env)
        self.__update_observation_space(env)

    def __update_observation_space(self, env):
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(3 * 2,), dtype='float32')

    def observation(self, obs):
        return np.concatenate([
            obs['robot_pos'],
            obs['obj_dist']
        ])


class KeepArmDownward(gym.Wrapper):
    """
    keep robot arm downward and disable uvw control
    Remove uvw from robot pose since they are fixed
    Remove gripper pose since we dont have this in real world
    """
    def __init__(self, env):
        gym.Wrapper.__init__(self, env)
        self.__update_spaces(env)

    def __update_spaces(self, env):
        self.action_space = gym.spaces.Box(-np.inf, np.inf, shape=(3,), dtype='float32')
        space_dict = copy.copy(env.observation_space.spaces)
        space_dict['robot_pos'] = gym.spaces.Box(-np.inf, np.inf, shape=(3,), dtype='float32')
        space_dict['gripper_pos'] = gym.spaces.Box(-np.inf, np.inf, shape=(3,), dtype='float32')
        space_dict['obj_dist'] = gym.spaces.Box(-np.inf, np.inf, shape=(3,), dtype='float32')
        self.observation_space = gym.spaces.Dict(space_dict)

    def reset(self, **kwargs):
        self.env.reset(**kwargs)
        self.unwrapped.robot.setJointPos([-90, 0, 0, 0, 90, 0], wait=True)  # set arm downward
        obs = self.unwrapped._get_obs()  # get new obs
        obs['robot_pos'] = obs['robot_pos'][:3]
        obs['gripper_pos'] = obs['gripper_pos'][:3]
        obs['obj_dist'] = obs['obj_dist'][:3]
        return obs

    def step(self, action):
        assert len(action) == 3, 'Only xyz control is allowed, action dim is %d != 3' % len(action)
        action = list(action) + [0, 0, 0]
        obs, reward, done, info = self.env.step(action)
        obs['robot_pos'] = obs['robot_pos'][:3]
        obs['gripper_pos'] = obs['gripper_pos'][:3]
        obs['obj_dist'] = obs['obj_dist'][:3]
        return obs, reward, done, info


class DiscretizeActionXYZ(gym.Wrapper):
    """
    discretize action in xyz dims
    take action levels in each direction as inputs
    make action incremental instead of absolute
    must be called after KeepArmDownward
    """
    def __init__(self, env, levels):
        gym.Wrapper.__init__(self, env)
        x_levels, y_levels, z_levels = levels['x'], levels['y'], levels['z']
        self.action_map = self.__create_action_map(x_levels, y_levels, z_levels)
        self.action_space = gym.spaces.Discrete(len(self.action_map))

    @staticmethod
    def __create_action_map(x, y, z):
        action_list = np.array(np.meshgrid(x, y, z), dtype=float).T.reshape(-1, 3)
        return dict(enumerate(action_list))

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    def step(self, action):
        action = copy.copy(self.action_map[action])
        return self.env.step(action)


class DeltaAction(gym.Wrapper):
    def __init__(self, env):
        gym.Wrapper.__init__(self, env)

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    def step(self, action):
        current_pose = self.unwrapped.robot.getRobotPos()
        action += current_pose
        return self.env.step(action)


class ClipAction(gym.Wrapper):
    def __init__(self, env):
        gym.Wrapper.__init__(self, env)
        self.max_action = 5
        self.action_space = gym.spaces.Box(-self.max_action, self.max_action,
                                           shape=self.action_space.shape, dtype='float32')

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    def step(self, action):
        action = np.clip(action, a_min=-self.max_action, a_max=self.max_action)
        return self.env.step(action)


class PenalizeMoveAwayReward(gym.Wrapper):
    """
    if the robot arm moves away from the object, get -1 reward

    """
    def __init__(self, env):
        gym.Wrapper.__init__(self, env)

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    def step(self, action):
        prev_pose = self.unwrapped.robot.getRobotPos()[:3]
        obs, reward, done, info = self.env.step(action)
        curr_pose = obs['robot_pos'][:3]
        obj_pose = info['object_pos'][:3]

        reward = -1.0
        if info['done_type'] > 1:  # for abnormal done
            reward += -10
        elif info['done_type'] == 1:  # for normal done
            reward += 20
        elif info['done_type'] == 0:  # for normal move
            curr_dist = np.abs(curr_pose - obj_pose)
            prev_dist = np.abs(prev_pose - obj_pose)
            diff = prev_dist - curr_dist
            mask = np.logical_not(np.isclose(diff, [0, 0, 0], atol=1.5))
            if np.any(mask):
                diff *= mask
                reward += np.sum(np.sign(diff))
            else:
                reward += -1
        return obs, reward, done, info


class ReacherReward(gym.Wrapper):
    def __init__(self, env, reward_type):
        gym.Wrapper.__init__(self, env)
        self._reward_type = reward_type

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        obj_dist = np.linalg.norm(obs['obj_dist'], ord=2) / 1000  # mm -> m
        if self._reward_type == "linear":
            reward_dist = -obj_dist
        elif self._reward_type == "precision":
            reward_dist = -obj_dist + np.exp(-obj_dist ** 2 / 0.01)
        elif self._reward_type == "precision_only":
            reward_dist = np.exp(-obj_dist ** 2 / 0.01)
        # print(reward_dist)

        return obs, reward_dist, done, info


def wrap_pick_place(env, obs_type, reward_type='linear', levels=None):
    env = DeltaAction(env)
    env = KeepArmDownward(env)
    env = ClipAction(env)

    if levels:
        env = DiscretizeActionXYZ(env, levels)
    env = ReacherReward(env, reward_type)

    if obs_type == 'image_only':
        env = WarpAndStackImages(env)
        env = RemoveRobotPose(env)
    elif obs_type == 'image_with_pose':
        env = WarpAndStackImages(env)
    elif obs_type == 'pose_only':
        env = NoImageObservation(env)

    return env


def plot_precision_reward():
    import matplotlib.pyplot as plt
    d = np.arange(0, 1, 0.01)
    r = np.exp(-d**2 / 0.01)
    plt.plot(d, r)
    plt.show()
