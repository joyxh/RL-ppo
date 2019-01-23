import os
import os.path as osp
import gym
import time
import joblib
import numpy as np
import tensorflow as tf
from common import logger

from common.misc_util import set_global_seeds
from common.math_util import explained_variance
from common import tf_util

from algorithms.ac.utils import discount_with_dones
from algorithms.ac.utils import Scheduler, make_path, find_trainable_variables
from algorithms.ac.utils import cat_entropy, mse


class Model(object):

    def __init__(self, policy, ob_space, ac_space, nenvs, nsteps,
            ent_coef=0.01, vf_coef=0.5, max_grad_norm=0.5, lr=7e-4,
            alpha=0.99, epsilon=1e-5, total_timesteps=int(80e6), lrschedule='linear'):

        sess = tf_util.make_session()
        nact = ac_space.n
        nbatch = nenvs*nsteps

        A = tf.placeholder(tf.int32, [nbatch])  # action
        ADV = tf.placeholder(tf.float32, [nbatch])  # advantage
        R = tf.placeholder(tf.float32, [nbatch])  # reward, 其实是根据样本估算出来的state value
        LR = tf.placeholder(tf.float32, [])  # 学习率

        step_model = policy(sess, ob_space, ac_space, nenvs, 1, reuse=False)  # 专门用于采样
        train_model = policy(sess, ob_space, ac_space, nenvs*nsteps, nsteps, reuse=True)  # 用于更新网络，与step_model共享主体网络

        # cross entropy loss of the correct action
        # 下面这句话做了很多事情，首先它把各个action的分数转换为了各个action的概率
        # 然后根据交叉熵，计算出了作为label的action的log probability，这就是策略梯度理论中的log(pi)
        # 然后加上一个负号，把梯度下降改为了梯度上升
        neglogpac = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=train_model.pi, labels=A)
        pg_loss = tf.reduce_mean(ADV * neglogpac)  # policy gradient loss log(pi)*adv

        vf_loss = tf.reduce_mean(mse(tf.squeeze(train_model.vf), R))  # value function mse loss
        entropy = tf.reduce_mean(cat_entropy(train_model.pi))  # 这里求的是策略pi的熵，我们要最大化这个熵
        loss = pg_loss - entropy*ent_coef + vf_loss * vf_coef  # total loss

        params = find_trainable_variables("model")  # 找到模型中可以被训练的tensor
        grads = tf.gradients(loss, params)  # dloss/dtheta
        if max_grad_norm is not None:
            grads, grad_norm = tf.clip_by_global_norm(grads, max_grad_norm)  # 限制最大的梯度
        grads = list(zip(grads, params))
        trainer = tf.train.RMSPropOptimizer(learning_rate=LR, decay=alpha, epsilon=epsilon)
        _train = trainer.apply_gradients(grads)  # 更新网络参数

        lr = Scheduler(v=lr, nvalues=total_timesteps, schedule=lrschedule)

        def train(obs, states, rewards, masks, actions, values):
            advs = rewards - values  # 计算advantage
            for step in range(len(obs)):
                cur_lr = lr.value()
            td_map = {train_model.X:obs, A:actions, ADV:advs, R:rewards, LR:cur_lr}
            if states is not None:
                td_map[train_model.S] = states
                td_map[train_model.M] = masks
            policy_loss, value_loss, policy_entropy, _ = sess.run(
                [pg_loss, vf_loss, entropy, _train],
                td_map
            )
            return policy_loss, value_loss, policy_entropy

        def save(save_path):
            ps = sess.run(params)
            make_path(osp.dirname(save_path))
            joblib.dump(ps, save_path)

        def load(load_path):
            loaded_params = joblib.load(load_path)
            restores = []
            for p, loaded_p in zip(params, loaded_params):
                restores.append(p.assign(loaded_p))
            ps = sess.run(restores)

        self.train = train
        self.train_model = train_model
        self.step_model = step_model
        self.step = step_model.step
        self.value = step_model.value
        self.initial_state = step_model.initial_state
        self.save = save
        self.load = load
        tf.global_variables_initializer().run(session=sess)


class Runner(object):

    def __init__(self, env, model, nsteps=5, gamma=0.99):
        self.env = env  # 这里的环境是通过多进程运行的多个环境
        self.model = model
        nh, nw, nc = env.observation_space.shape
        nenv = env.num_envs  # 环境的数量，也就是进程的数量
        self.batch_ob_shape = (nenv*nsteps, nh, nw, nc)
        self.obs = np.zeros((nenv, nh, nw, nc), dtype=np.uint8)
        self.nc = nc
        obs = env.reset()  # 这里是个bug，初始化obs永远为0；另外此处reset可以获得多个环境的obs
        self.gamma = gamma
        self.nsteps = nsteps
        self.states = model.initial_state
        self.dones = [False for _ in range(nenv)]  # 将所有环境的done初始化为0
        self.episode_returns = []
        self.single_episode_returns = np.zeros(nenv)

    def run(self):
        mb_obs, mb_rewards, mb_actions, mb_values, mb_dones = [],[],[],[],[]
        mb_states = self.states
        for n in range(self.nsteps):  # 每次都采集固定的步数,猜测因为这样方便同步多进程
            # 如果只考慮CNN策略，那麼下面Step裏面的states, dones是沒用的，他們用於LSTM
            # 将所有环境的obs丢进网络，并获得每个环境的action
            actions, values, states, _ = self.model.step(self.obs, self.states, self.dones)
            mb_obs.append(np.copy(self.obs))
            mb_actions.append(actions)
            mb_values.append(values)
            mb_dones.append(self.dones)
            # step所有环境，并获得每个环境的obs
            # 注意，当done=1时，此处的step会自动reset并返回新的obs
            obs, rewards, dones, _ = self.env.step(actions)
            self.single_episode_returns += rewards
            for i in range(self.env.num_envs):
                if dones[i]:
                    self.episode_returns.append(self.single_episode_returns[i])
                    self.single_episode_returns[i] = 0
            self.states = states
            self.dones = dones
            for n, done in enumerate(dones):
                if done:
                    self.obs[n] = self.obs[n]*0
            # 因为下面这句话，上面这个循环似乎没有用
            self.obs = obs
            mb_rewards.append(rewards)
        mb_dones.append(self.dones)
        #batch of steps to batch of rollouts
        mb_obs = np.asarray(mb_obs, dtype=np.uint8).swapaxes(1, 0).reshape(self.batch_ob_shape)
        mb_rewards = np.asarray(mb_rewards, dtype=np.float32).swapaxes(1, 0)
        mb_actions = np.asarray(mb_actions, dtype=np.int32).swapaxes(1, 0)
        mb_values = np.asarray(mb_values, dtype=np.float32).swapaxes(1, 0)
        mb_dones = np.asarray(mb_dones, dtype=np.bool).swapaxes(1, 0)
        # 下面这一个去头一个去尾是什么意思
        mb_masks = mb_dones[:, :-1]
        mb_dones = mb_dones[:, 1:]  # 因为第一个dones在env.step之前就被添加进去了，所以需要去掉

        # 获得最后一个step之后的V估计
        last_values = self.model.value(self.obs, self.states, self.dones).tolist()
        #discount/bootstrap off value fn
        for n, (rewards, dones, value) in enumerate(zip(mb_rewards, mb_dones, last_values)):
            rewards = rewards.tolist()
            dones = dones.tolist()
            # 如果最后一个done为0，也就是说这个episode没有结束
            # 那么需要使用last_values这个预测的V来进行bootstrap计算
            # 否则使用MC方法即可
            if dones[-1] == 0:
                rewards = discount_with_dones(rewards+[value], dones+[0], self.gamma)[:-1]
            else:
                rewards = discount_with_dones(rewards, dones, self.gamma)
            mb_rewards[n] = rewards

        mb_rewards = mb_rewards.flatten()
        mb_actions = mb_actions.flatten()
        mb_values = mb_values.flatten()
        mb_masks = mb_masks.flatten()

        return mb_obs, mb_states, mb_rewards, mb_masks, mb_actions, mb_values, self.episode_returns


def learn(policy, env, seed, nsteps=5, total_timesteps=int(80e6), vf_coef=0.5, ent_coef=0.01, max_grad_norm=0.5,
          lr=7e-4, lrschedule='linear', epsilon=1e-5, alpha=0.99, gamma=0.99, log_interval=100):
    tf.reset_default_graph()
    set_global_seeds(seed)

    nenvs = env.num_envs  # 环境数量
    ob_space = env.observation_space  # observation shape
    ac_space = env.action_space     # action 数量
    # 初始化网络更新模块
    model = Model(policy=policy, ob_space=ob_space, ac_space=ac_space, nenvs=nenvs, nsteps=nsteps,
                  ent_coef=ent_coef, vf_coef=vf_coef, max_grad_norm=max_grad_norm, lr=lr, alpha=alpha,
                  epsilon=epsilon, total_timesteps=total_timesteps, lrschedule=lrschedule)
    # 初始化采样模块
    runner = Runner(env, model, nsteps=nsteps, gamma=gamma)

    nbatch = nenvs*nsteps  # batchsize
    tstart = time.time()
    mean_100ep_reward = 0
    for update in range(1, total_timesteps//nbatch+1):
        obs, states, rewards, masks, actions, values, episode_returns = runner.run()  # 采样
        policy_loss, value_loss, policy_entropy = model.train(obs, states, rewards, masks, actions, values)  # 更新网络
        nseconds = time.time()-tstart
        fps = int((update*nbatch)/nseconds)
        num_episodes = len(episode_returns)
        if num_episodes >= 100:
            mean_100ep_reward = round(np.mean(episode_returns[-101:-1]), 1)
        if update % 10000 == 0:
            model_path = os.path.join(logger.get_dir(), 'model.pkl')
            logger.info('Dumping model to ', model_path)
            model.save(model_path)
        if update % log_interval == 0 or update == 1:
            ev = explained_variance(values, rewards)
            logger.record_tabular("nupdates", update)
            logger.record_tabular("total_timesteps", update*nbatch)
            logger.record_tabular("fps", fps)
            logger.record_tabular("policy_entropy", float(policy_entropy))
            logger.record_tabular("value_loss", float(value_loss))
            logger.record_tabular("explained_variance", float(ev))
            logger.record_tabular("mean_update_reward", np.mean(rewards))
            logger.record_tabular("mean 100 episode reward", mean_100ep_reward)
            logger.record_tabular("number of episodes", num_episodes)
            logger.dump_tabular()
    env.close()
