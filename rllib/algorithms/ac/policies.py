import numpy as np
import tensorflow as tf
from algorithms.ac.utils import conv, fc, conv_to_fc, batch_to_seq, seq_to_batch, lstm, lnlstm
from common.distributions import make_pdtype
import os
import joblib


def nature_cnn(unscaled_images):
    """
    CNN from Nature paper.
    """
    scaled_images = tf.cast(unscaled_images, tf.float32) / 255.
    activ = tf.nn.relu
    h = activ(conv(scaled_images, 'c1', nf=32, rf=8, stride=4, init_scale=np.sqrt(2)))
    h2 = activ(conv(h, 'c2', nf=64, rf=4, stride=2, init_scale=np.sqrt(2)))
    h3 = activ(conv(h2, 'c3', nf=64, rf=3, stride=1, init_scale=np.sqrt(2)))
    h3 = conv_to_fc(h3)
    return activ(fc(h3, 'fc1', nh=512, init_scale=np.sqrt(2)))


class LnLstmPolicy(object):
    def __init__(self, sess, ob_space, ac_space, nbatch, nsteps, nlstm=256, reuse=False):
        nenv = nbatch // nsteps
        nh, nw, nc = ob_space.shape
        ob_shape = (nbatch, nh, nw, nc)
        nact = ac_space.n
        X = tf.placeholder(tf.uint8, ob_shape) #obs
        M = tf.placeholder(tf.float32, [nbatch]) #mask (done t-1)
        S = tf.placeholder(tf.float32, [nenv, nlstm*2]) #states
        with tf.variable_scope("model", reuse=reuse):
            h = nature_cnn(X)
            xs = batch_to_seq(h, nenv, nsteps)
            ms = batch_to_seq(M, nenv, nsteps)
            h5, snew = lnlstm(xs, ms, S, 'lstm1', nh=nlstm)
            h5 = seq_to_batch(h5)
            pi = fc(h5, 'pi', nact)
            vf = fc(h5, 'v', 1)

        self.pdtype = make_pdtype(ac_space)
        self.pd = self.pdtype.pdfromflat(pi)

        v0 = vf[:, 0]
        a0 = self.pd.sample()
        neglogp0 = self.pd.neglogp(a0)
        self.initial_state = np.zeros((nenv, nlstm*2), dtype=np.float32)

        def step(ob, state, mask):
            return sess.run([a0, v0, snew, neglogp0], {X:ob, S:state, M:mask})

        def value(ob, state, mask):
            return sess.run(v0, {X:ob, S:state, M:mask})

        self.X = X
        self.M = M
        self.S = S
        self.pi = pi
        self.vf = vf
        self.step = step
        self.value = value


class LstmPolicy(object):

    def __init__(self, sess, ob_space, ac_space, nbatch, nsteps, nlstm=256, reuse=False):
        nenv = nbatch // nsteps

        nh, nw, nc = ob_space.shape
        ob_shape = (nbatch, nh, nw, nc)
        nact = ac_space.n
        X = tf.placeholder(tf.uint8, ob_shape) #obs
        M = tf.placeholder(tf.float32, [nbatch]) #mask (done t-1)
        S = tf.placeholder(tf.float32, [nenv, nlstm*2]) #states
        with tf.variable_scope("model", reuse=reuse):
            h = nature_cnn(X)
            xs = batch_to_seq(h, nenv, nsteps)
            ms = batch_to_seq(M, nenv, nsteps)
            h5, snew = lstm(xs, ms, S, 'lstm1', nh=nlstm)
            h5 = seq_to_batch(h5)
            pi = fc(h5, 'pi', nact)
            vf = fc(h5, 'v', 1)

        self.pdtype = make_pdtype(ac_space)
        self.pd = self.pdtype.pdfromflat(pi)

        v0 = vf[:, 0]
        a0 = self.pd.sample()
        neglogp0 = self.pd.neglogp(a0)
        self.initial_state = np.zeros((nenv, nlstm*2), dtype=np.float32)

        def step(ob, state, mask):
            return sess.run([a0, v0, snew, neglogp0], {X:ob, S:state, M:mask})

        def value(ob, state, mask):
            return sess.run(v0, {X:ob, S:state, M:mask})

        self.X = X
        self.M = M
        self.S = S
        self.pi = pi
        self.vf = vf
        self.step = step
        self.value = value


class Policy(object):
    def __init__(self, sess):
        self.sess = sess

    def save(self, _dir):
        trainable_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='model')
        model_params = self.sess.run(trainable_variables)
        save_path = os.path.join(_dir, 'model')
        joblib.dump(model_params, save_path)

    def load(self, _dir):
        load_path = os.path.join(_dir, 'model')
        if os.path.exists(load_path):
            loaded_params = joblib.load(load_path)
            trainable_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='model')
            restores = []
            for p, loaded_p in zip(trainable_variables, loaded_params):
                restores.append(p.assign(loaded_p))
            self.sess.run(restores)
            print('Loaded model parameters from', load_path)
            return True
        else:
            print('Could not find model file', load_path)
            return False


class CnnPolicyV2(Policy):
    def __init__(self, sess, ob_space, ac_space, reuse=False):
        super().__init__(sess)
        h, w, c = ob_space.shape
        obs_ph = tf.placeholder(tf.uint8, shape=[None, h, w, c], name='observation_input')

        with tf.variable_scope('model', reuse=reuse):
            backbone = nature_cnn(obs_ph)  # 主体部分共享
            pi = fc(backbone, 'policy', ac_space.n, init_scale=0.01)  # 策略头，输出为各个action的分数
            vf = fc(backbone, 'value', 1)[:, 0]  # 值函数头，输出为一个值

        with tf.variable_scope('policy_distribution'):
            # 下面是将各个action的分数转换为一个概率分布
            # 只会在采样的时候用到，更新网络时的损失函数另外设计
            self.pdtype = make_pdtype(ac_space)
            self.pd = self.pdtype.pdfromflat(pi)

            a0 = self.pd.sample()  # 根据概率分布随机采一个样本
            neglogp0 = self.pd.neglogp(a0)  # -log(a0)

        def step(obs):
            # 采一步样本
            # 输入为图片
            # 输出为action, value以及-log(action)
            return sess.run(fetches=[a0, vf, neglogp0], feed_dict={obs_ph: obs})

        def value(obs):
            # 根据输入图片返回神经网络预测的状态值
            return sess.run(fetches=vf, feed_dict={obs_ph: obs})

        self.pi = pi
        self.vf = vf
        self.obs_ph = obs_ph
        self.step = step
        self.value = value


class CnnPolicy(object):

    def __init__(self, sess, ob_space, ac_space, nbatch, nsteps, reuse=False): #pylint: disable=W0613
        nh, nw, nc = ob_space.shape  # 输入图片的大小
        ob_shape = (nbatch, nh, nw, nc)
        nact = ac_space.n   # action的数量
        X = tf.placeholder(tf.uint8, ob_shape) #obs
        with tf.variable_scope("model", reuse=reuse):
            h = nature_cnn(X)  # 主体部分共享
            pi = fc(h, 'pi', nact, init_scale=0.01)  # 策略头，输出为各个action的分数
            vf = fc(h, 'v', 1)[:,0]  # 值函数头，输出为一个值

        # 下面是将各个action的分数转换为一个概率分布
        # 只会在采样的时候用到，更新网络时的损失函数另外设计
        self.pdtype = make_pdtype(ac_space)
        self.pd = self.pdtype.pdfromflat(pi)

        a0 = self.pd.sample()  # 根据概率分布随机采一个样本
        neglogp0 = self.pd.neglogp(a0)  # -log(a0)
        self.initial_state = None

        def step(ob, *_args, **_kwargs):
            # 采一步样本
            # 输入为图片
            # 输出为action, value以及-log(action)
            # self.initial_state只对LSTM有用
            a, v, neglogp = sess.run([a0, vf, neglogp0], {X:ob})
            return a, v, self.initial_state, neglogp

        def value(ob, *_args, **_kwargs):
            # 根据输入图片返回神经网络预测的状态值
            return sess.run(vf, {X:ob})

        self.X = X
        self.pi = pi
        self.vf = vf
        self.step = step
        self.value = value


class MlpPolicy(object):
    def __init__(self, sess, ob_space, ac_space, nbatch, nsteps, reuse=False): #pylint: disable=W0613
        ob_shape = (nbatch,) + ob_space.shape
        actdim = ac_space.shape[0]
        X = tf.placeholder(tf.float32, ob_shape, name='Ob') #obs
        with tf.variable_scope("model", reuse=reuse):
            activ = tf.tanh
            h1 = activ(fc(X, 'pi_fc1', nh=64, init_scale=np.sqrt(2)))
            h2 = activ(fc(h1, 'pi_fc2', nh=64, init_scale=np.sqrt(2)))
            pi = fc(h2, 'pi', actdim, init_scale=0.01)
            h1 = activ(fc(X, 'vf_fc1', nh=64, init_scale=np.sqrt(2)))
            h2 = activ(fc(h1, 'vf_fc2', nh=64, init_scale=np.sqrt(2)))
            vf = fc(h2, 'vf', 1)[:,0]
            logstd = tf.get_variable(name="logstd", shape=[1, actdim],
                initializer=tf.zeros_initializer())

        pdparam = tf.concat([pi, pi * 0.0 + logstd], axis=1)

        self.pdtype = make_pdtype(ac_space)
        self.pd = self.pdtype.pdfromflat(pdparam)

        a0 = self.pd.sample()
        neglogp0 = self.pd.neglogp(a0)
        self.initial_state = None

        def step(ob, *_args, **_kwargs):
            a, v, neglogp = sess.run([a0, vf, neglogp0], {X:ob})
            return a, v, self.initial_state, neglogp

        def value(ob, *_args, **_kwargs):
            return sess.run(vf, {X:ob})

        self.X = X
        self.pi = pi
        self.vf = vf
        self.step = step
        self.value = value
