#!/usr/bin/env python3
import os
import yaml
import pprint
import argparse
from datetime import datetime

import gym

from common import logger, Monitor
from common.misc_util import set_global_seeds
from algorithms.ppo1 import cnn_policy, mlp_policy, ppo_single_process

from envs import wrappers
import envs
import sys


DEFAULT_LOG_DIR = os.path.join('experiments', 'ppo1_single2')
env_config_dir = 'config/reacher_env_config.yaml'

if not os.path.exists(DEFAULT_LOG_DIR):
    os.makedirs(DEFAULT_LOG_DIR)


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--seed', help='random seed', default=0, type=int)
    parser.add_argument('--exp-id', help='experiment id', default=None)
    parser.add_argument('--env-id', help='environment id', default='SpaceInvadersNoFrameskip-v4')
    parser.add_argument('--config-file', help='config file path', default='config/trpo_default.yaml')
    args = parser.parse_args()

    if args.exp_id is None:
        log_dir = os.path.join(DEFAULT_LOG_DIR, args.env_id, datetime.now().strftime('%Y%m%d-%H%M'))
        os.makedirs(log_dir, exist_ok=True)
        config = yaml.load(open(os.path.join(args.config_file)))
        config['seed'] = args.seed
        yaml.dump(config, open(os.path.join(log_dir, 'CONFIG'), 'w'))
        env_config = yaml.load(open(env_config_dir))
        yaml.dump(env_config, open(os.path.join(log_dir, 'ENV_CONFIG'), 'w'))

    else:
        log_dir = os.path.join(DEFAULT_LOG_DIR, args.env_id, args.exp_id)
        config = yaml.load(open(os.path.join(log_dir, 'CONFIG')))

    pprint.pprint(config)
    logger.configure(log_dir, ['stdout', 'tensorboard', 'log', 'csv'])

    set_global_seeds(args.seed)
    if args.env_id.startswith('PickPlace'):
        env = gym.make(args.env_id)
        env = wrappers.wrap_pick_place(env, obs_type=config['obs_type'], reward_type=config['reward_type'])
        env = Monitor(env, log_dir)
    elif args.env_id.startswith('Reacher'):
        env = gym.make(args.env_id)
        env = wrappers.wrap_reacher(env)
        env = Monitor(env, log_dir)
    else:
        env = wrappers.make_atari(args.env_id)
        env = Monitor(env, log_dir)
        env = wrappers.wrap_deepmind(env, frame_stack=True)

    def policy_fn(name, ob_space, ac_space):
        policy_config = config['policy']
        if config['obs_type'] == 'image_only':
            return cnn_policy.CnnPolicy(name=name, ob_space=ob_space, ac_space=ac_space, kind='small')
        elif config['obs_type'] == 'image_with_pose':
            return cnn_policy.CnnPolicyDict(name=name, ob_space=env.observation_space, ac_space=env.action_space)
        elif config['obs_type'] == 'pose_only':
            return mlp_policy.MlpPolicy(name=name, ob_space=ob_space, ac_space=ac_space,
                                  hid_size=policy_config['hidden_size'],
                                  num_hid_layers=policy_config['hidden_layers'],
                                  gaussian_fixed_var=policy_config['gaussian_fixed_var'])

    ppo_single_process.learn(env, policy_fn,
                             max_timesteps=config['num_timesteps'],
                             timesteps_per_actorbatch=config['timesteps_per_actorbatch'],
                             clip_param=config['clip_param'],
                             entcoeff=config['entcoeff'],
                             optim_epochs=config['optim_epochs'],
                             optim_stepsize=config['optim_stepsize'],
                             optim_batchsize=config['optim_batchsize'],
                             gamma=config['gamma'],
                             lam=config['lam'],
                             schedule=config['schedule']
                             )
    env.close()


if __name__ == "__main__":
    main()
