import os
import time
import tempfile

import pickle
import tensorflow as tf
import zipfile
import cloudpickle
import numpy as np

import gym
import common.tf_util as U
from common import logger
from common.schedules import LinearSchedule
from algorithms.deepq.build_graph import build_act, build_train
from algorithms.deepq.replay_buffer import ReplayBuffer, PrioritizedReplayBuffer
from algorithms.deepq.utils import load_state, save_state


class ActWrapper(object):
    def __init__(self, act, act_params):
        self._act = act
        self._act_params = act_params

    @staticmethod
    def load(path):
        with open(path, "rb") as f:
            model_data, act_params = cloudpickle.load(f)
        act = build_act(**act_params)
        sess = tf.Session()
        sess.__enter__()
        with tempfile.TemporaryDirectory() as td:
            arc_path = os.path.join(td, "packed.zip")
            with open(arc_path, "wb") as f:
                f.write(model_data)

            zipfile.ZipFile(arc_path, 'r', zipfile.ZIP_DEFLATED).extractall(td)
            load_state(os.path.join(td, "model"))

        return ActWrapper(act, act_params)

    def __call__(self, *args, **kwargs):
        return self._act(*args, **kwargs)

    def save(self, path=None):
        """Save model to a pickle located at `path`"""
        if path is None:
            path = os.path.join(logger.get_dir(), "model.pkl")

        with tempfile.TemporaryDirectory() as td:
            save_state(os.path.join(td, "model"))
            arc_name = os.path.join(td, "packed.zip")
            with zipfile.ZipFile(arc_name, 'w') as zipf:
                for root, dirs, files in os.walk(td):
                    for fname in files:
                        file_path = os.path.join(root, fname)
                        if file_path != arc_name:
                            zipf.write(file_path, os.path.relpath(file_path, td))
            with open(arc_name, "rb") as f:
                model_data = f.read()
        with open(path, "wb") as f:
            cloudpickle.dump((model_data, self._act_params), f)


def load(path):
    """Load act function that was returned by learn function.

    Parameters
    ----------
    path: str
        path to the act function pickle

    Returns
    -------
    act: ActWrapper
        function that takes a batch of observations
        and returns actions.
    """
    return ActWrapper.load(path)


def learn(env,
          q_func,
          make_obs_ph,
          lr=5e-4,
          max_timesteps=100000,
          buffer_size=50000,
          exploration_fraction=0.1,
          exploration_final_eps=0.02,
          train_freq=1,
          batch_size=32,
          print_freq=100,
          checkpoint_freq=10000,
          learning_starts=1000,
          gamma=1.0,
          target_network_update_freq=500,
          prioritized_replay=False,
          prioritized_replay_alpha=0.6,
          prioritized_replay_beta0=0.4,
          prioritized_replay_beta_iters=None,
          prioritized_replay_eps=1e-6,
          param_noise=False,
          callback=None,
          replay_buffer=ReplayBuffer,
          ):
    """Train a deepq model.

    Parameters
    -------
    env: gym.Env
        environment to train on
    q_func: (tf.Variable, int, str, bool) -> tf.Variable
        the model that takes the following inputs:
            observation_in: object
                the output of observation placeholder
            num_actions: int
                number of actions
            scope: str
            reuse: bool
                should be passed to outer variable scope
        and returns a tensor of shape (batch_size, num_actions) with values of every action.
    lr: float
        learning rate for adam optimizer
    max_timesteps: int
        number of env steps to optimizer for
    buffer_size: int
        size of the replay buffer
    exploration_fraction: float
        fraction of entire training period over which the exploration rate is annealed
    exploration_final_eps: float
        final value of random action probability
    train_freq: int
        update the model every `train_freq` steps.
        set to None to disable printing
    batch_size: int
        size of a batched sampled from replay buffer for training
    print_freq: int
        how often to print out training progress
        set to None to disable printing
    checkpoint_freq: int
        how often to save the model. This is so that the best version is restored
        at the end of the training. If you do not wish to restore the best version at
        the end of the training set this variable to None.
    learning_starts: int
        how many steps of the model to collect transitions for before learning starts
    gamma: float
        discount factor
    target_network_update_freq: int
        update the target network every `target_network_update_freq` steps.
    prioritized_replay: True
        if True prioritized replay buffer will be used.
    prioritized_replay_alpha: float
        alpha parameter for prioritized replay buffer
    prioritized_replay_beta0: float
        initial value of beta for prioritized replay buffer
    prioritized_replay_beta_iters: int
        number of iterations over which beta will be annealed from initial value
        to 1.0. If set to None equals to max_timesteps.
    prioritized_replay_eps: float
        epsilon to add to the TD errors when updating priorities.
    callback: (locals, globals) -> None
        function called at every steps with state of the algorithm.
        If callback returns true training stops.

    Returns
    -------
    act: ActWrapper
        Wrapper over act function. Adds ability to save it and load it.
        See header of baselines/deepq/categorical.py for details on the act function.
    """
    # Create all the functions necessary to train the model
    
    # 这两句话似乎相当于是with tf.Session（） as sess:
    sess = tf.Session()
    sess.__enter__()

    act, train, update_target, debug = build_train(
        make_obs_ph=make_obs_ph,
        q_func=q_func,
        num_actions=env.action_space.n,
        optimizer=tf.train.AdamOptimizer(learning_rate=lr),
        gamma=gamma,
        grad_norm_clipping=10,
        param_noise=param_noise
    )

    act_params = {
        'make_obs_ph': make_obs_ph,
        'q_func': q_func,
        'num_actions': env.action_space.n,
    }

    act = ActWrapper(act, act_params)

    # Create the replay buffer
    if prioritized_replay:
        replay_buffer = PrioritizedReplayBuffer(buffer_size, alpha=prioritized_replay_alpha)
        if prioritized_replay_beta_iters is None:
            prioritized_replay_beta_iters = max_timesteps
        beta_schedule = LinearSchedule(prioritized_replay_beta_iters,
                                       initial_p=prioritized_replay_beta0,
                                       final_p=1.0)
    else:
        # replay_buffer = ReplayBuffer(buffer_size)
        replay_buffer = replay_buffer(buffer_size)
        beta_schedule = None

    checkpoint_dir = os.path.join(logger.get_dir(), 'checkpoint')
    # 加载replay buffer
    if os.path.exists(checkpoint_dir):
        logger.info('Loading replay buffer from', checkpoint_dir)
        replay_buffer.load_state(checkpoint_dir)

    # Create the schedule for exploration starting from 1.
    exploration = LinearSchedule(schedule_timesteps=int(exploration_fraction * max_timesteps),
                                 initial_p=1.0,
                                 final_p=exploration_final_eps)

    # Initialize the parameters and copy them to the target network.
    U.initialize()
    # 同步behaviour q和target q
    update_target()

    # 初始化或加载循环参数
    loop_params = {
        'episode_rewards': [0.0],
        'saved_mean_reward': None,
        'next_t': 0,
        'total_time': 0,
        'total_sampling_time': 0,
        'total_training_time': 0,
    }
    loop_params_file = os.path.join(checkpoint_dir, 'loop_params.pkl')
    if os.path.exists(loop_params_file):
        logger.info('Loading loop parameters from', loop_params_file)
        loop_params.update((pickle.load(open(loop_params_file, 'rb'))))

    episode_rewards = loop_params['episode_rewards']
    saved_mean_reward = loop_params['saved_mean_reward']
    t_start = loop_params['next_t']
    tt_t = loop_params['total_time']
    ts_t = loop_params['total_sampling_time']
    ttr_t = loop_params['total_training_time']

    obs = env.reset()
    reset = True

    model_saved = False
    # 加载tf checkpoint
    model_file = os.path.join(checkpoint_dir, "model")
    if tf.train.latest_checkpoint(checkpoint_dir) is not None:
        logger.info('Loading model from {}'.format(model_file))
        load_state(model_file)

    for t in range(t_start, max_timesteps):
        tt_b = time.time()
        if callback is not None:
            if callback(locals(), globals()):
                break
        # Take action and update exploration to the newest value
        kwargs = {}
        if not param_noise:
            update_eps = exploration.value(t)
            update_param_noise_threshold = 0.
        else:
            update_eps = 0.
            # Compute the threshold such that the KL divergence between perturbed and non-perturbed
            # policy is comparable to eps-greedy exploration with eps = exploration.value(t).
            # See Appendix C.1 in Parameter Space Noise for Exploration, Plappert et al., 2017
            # for detailed explanation.
            update_param_noise_threshold = -np.log(1. - exploration.value(t) + exploration.value(t) / float(env.action_space.n))
            kwargs['reset'] = reset
            kwargs['update_param_noise_threshold'] = update_param_noise_threshold
            kwargs['update_param_noise_scale'] = True
        # 采样
        if isinstance(obs, dict):
            _obs = {}
            for key, value in obs.items():
                _obs[key] = np.array(obs[key])[None]
            action = act(_obs, update_eps=update_eps, **kwargs)[0]
        else:
            action = act(np.array(obs)[None], update_eps=update_eps, **kwargs)[0]
        env_action = action
        reset = False
        ts_b = time.time()
        new_obs, rew, done, _ = env.step(env_action)
        ts_t += time.time() - ts_b
        # Store transition in the replay buffer.
        replay_buffer.add(obs, action, rew, new_obs, float(done))
        obs = new_obs

        # 这里的reward应该是clip过了的
        episode_rewards[-1] += rew
        if done:
            obs = env.reset()
            episode_rewards.append(0.0)
            reset = True

        # 训练
        if t > learning_starts and t % train_freq == 0:
            # 每隔train_freq个step才会训练一次behaviour q
            # Minimize the error in Bellman's equation on a batch sampled from replay buffer.
            if prioritized_replay:
                experience = replay_buffer.sample(batch_size, beta=beta_schedule.value(t))
                (obses_t, actions, rewards, obses_tp1, dones, weights, batch_idxes) = experience
            else:
                obses_t, actions, rewards, obses_tp1, dones = replay_buffer.sample(batch_size)
                weights, batch_idxes = np.ones_like(rewards), None
            ttr_b = time.time()
            td_errors = train(obses_t, actions, rewards, obses_tp1, dones, weights)
            ttr_t += time.time() - ttr_b
            if prioritized_replay:
                new_priorities = np.abs(td_errors) + prioritized_replay_eps
                replay_buffer.update_priorities(batch_idxes, new_priorities)

        # 每隔target_network_update_freq个step才会同步一次behvaiour q和target q
        if t > learning_starts and t % target_network_update_freq == 0:
            # Update target network periodically.
            update_target()

        # 日志
        num_episodes = len(episode_rewards)
        mean_100ep_reward = round(np.mean(episode_rewards[-101:-1]), 1)
        if done and print_freq is not None and len(episode_rewards) % print_freq == 0:
            logger.record_tabular("steps", t)
            logger.record_tabular("episodes", num_episodes)
            logger.record_tabular("mean 100 episode reward", mean_100ep_reward)
            logger.record_tabular("% time spent exploring", int(100 * exploration.value(t)))
            logger.record_tabular("total time", tt_t)
            logger.record_tabular("total sampling time", ts_t)
            logger.record_tabular("total training time", ttr_t)
            logger.record_tabular("average total time", tt_t / float(t))
            logger.record_tabular("average sampling time", ts_t / float(t))
            logger.record_tabular("average training time", ttr_t / float(t))
            logger.dump_tabular()

        # checkpoints
        if checkpoint_freq is not None and t > learning_starts and t % checkpoint_freq == 0:
            logger.info('Dumping checkpoints...')
            save_state(model_file)  # tf models
            replay_buffer.save_state(checkpoint_dir)  # replay buffer
            loop_params.update({
                'episode_rewards': episode_rewards,
                'saved_mean_reward': saved_mean_reward,
                'next_t': t + 1,
                'total_time': tt_t,
                'total_sampling_time': ts_t,
                'total_training_time': ttr_t,
            })
            pickle.dump(loop_params, open(loop_params_file, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)  # loop parameters

        # 保存最好的模型
        if t > learning_starts and (saved_mean_reward is None or mean_100ep_reward > saved_mean_reward):
            if print_freq is not None:
                logger.log("Saving model due to mean reward increase: {} -> {}".format(
                           saved_mean_reward, mean_100ep_reward))
            act.save(os.path.join(logger.get_dir(), 'best_model_for_act'))
            model_saved = True
            saved_mean_reward = mean_100ep_reward

        tt_t += time.time() - tt_b

    # 训练结束时的模型不一定比之前保存下来的模型好
    # 所以这一步会重新读取之前保存下来的最好的模型
    # 然后通过act把这个最好的模型返回出去
    # if model_saved:
    #     if print_freq is not None:
    #         logger.log("Restored model with mean reward: {}".format(saved_mean_reward))
    #     load_state(model_file)

    return act
