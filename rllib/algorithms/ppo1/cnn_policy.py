import common.tf_util as U
import tensorflow as tf
import gym
from common.distributions import make_pdtype
import joblib
from algorithms.deepq.utils import DictTfInput

class CnnPolicy(object):
    recurrent = False

    def __init__(self, name, ob_space, ac_space, kind='large'):
        with tf.variable_scope(name):
            self._init(ob_space, ac_space, kind)
            self.scope = tf.get_variable_scope().name

    def _init(self, ob_space, ac_space, kind):
        assert isinstance(ob_space, gym.spaces.Box)

        self.pdtype = pdtype = make_pdtype(ac_space)
        sequence_length = None

        ob = U.get_placeholder(name="ob", dtype=tf.float32, shape=[sequence_length] + list(ob_space.shape))

        x = ob / 255.0
        if kind == 'small': # from A3C paper
            x = tf.nn.relu(U.conv2d(x, 16, "l1", [8, 8], [4, 4], pad="VALID"))
            x = tf.nn.relu(U.conv2d(x, 32, "l2", [4, 4], [2, 2], pad="VALID"))
            x = U.flattenallbut0(x)
            x = tf.nn.relu(tf.layers.dense(x, 256, name='lin', kernel_initializer=U.normc_initializer(1.0)))
        elif kind == 'large': # Nature DQN
            x = tf.nn.relu(U.conv2d(x, 32, "l1", [8, 8], [4, 4], pad="VALID"))
            x = tf.nn.relu(U.conv2d(x, 64, "l2", [4, 4], [2, 2], pad="VALID"))
            x = tf.nn.relu(U.conv2d(x, 64, "l3", [3, 3], [1, 1], pad="VALID"))
            x = U.flattenallbut0(x)
            x = tf.nn.relu(tf.layers.dense(x, 512, name='lin', kernel_initializer=U.normc_initializer(1.0)))
        else:
            raise NotImplementedError

        logits = tf.layers.dense(x, pdtype.param_shape()[0], name='logits', kernel_initializer=U.normc_initializer(0.01))
        self.pd = pdtype.pdfromflat(logits)
        self.vpred = tf.layers.dense(x, 1, name='value', kernel_initializer=U.normc_initializer(1.0))[:,0]

        self.state_in = []
        self.state_out = []

        stochastic = tf.placeholder(dtype=tf.bool, shape=())
        ac = self.pd.sample() # XXX
        self._act = U.function([stochastic, ob], [ac, self.vpred])

    def act(self, stochastic, ob):
        ac1, vpred1 = self._act(stochastic, ob[None])
        return ac1[0], vpred1[0]
    def get_variables(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.scope)
    def get_trainable_variables(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)
    def get_initial_state(self):
        return []
    def save(self, session, path):
        ps = session.run(self.get_variables())
        joblib.dump(ps, path)
    def load(self, session, path):
        loaded_params = joblib.load(path)
        restores = []
        for p, loaded_p in zip(self.get_variables(), loaded_params):
            restores.append(p.assign(loaded_p))
        session.run(restores)


class CnnPolicyDict(object):
    def __init__(self, name, ob_space, ac_space):
        with tf.variable_scope(name):
            self._init(ob_space, ac_space)
            self.scope = tf.get_variable_scope().name

    def _init(self, ob_space, ac_space):
        self.pdtype = pdtype = make_pdtype(ac_space)

        ob_spaces = ob_space.spaces

        image_ph = U.get_placeholder(name='image', dtype=tf.float32, shape=[None] + list(ob_spaces['image'].shape))
        robot_pos_ph = U.get_placeholder(name='robot_pos', dtype=tf.float32,
                                         shape=[None] + list(ob_spaces['robot_pos'].shape))

        ob = DictTfInput(None, ob_spaces,
                                   {
                                    'image': image_ph,
                                    'robot_pos': robot_pos_ph
                                   })


        obscaled = image_ph / 255.0
        robot_pos_ph /= 1000.0

        with tf.variable_scope("pol"):
            x = obscaled
            x = tf.nn.relu(U.conv2d(x, 8, "l1", [8, 8], [4, 4], pad="VALID"))
            x = tf.nn.relu(U.conv2d(x, 16, "l2", [4, 4], [2, 2], pad="VALID"))
            x = U.flattenallbut0(x)
            x = tf.concat([x, robot_pos_ph], axis=1)
            x = tf.nn.tanh(tf.layers.dense(x, 128, name='lin', kernel_initializer=U.normc_initializer(1.0)))
            logits = tf.layers.dense(x, pdtype.param_shape()[0], name='logits', kernel_initializer=U.normc_initializer(0.01))
            self.pd = pdtype.pdfromflat(logits)
        with tf.variable_scope("vf"):
            x = obscaled
            x = tf.nn.relu(U.conv2d(x, 8, "l1", [8, 8], [4, 4], pad="VALID"))
            x = tf.nn.relu(U.conv2d(x, 16, "l2", [4, 4], [2, 2], pad="VALID"))
            x = U.flattenallbut0(x)
            x = tf.concat([x, robot_pos_ph], axis=1)
            x = tf.nn.tanh(tf.layers.dense(x, 128, name='lin', kernel_initializer=U.normc_initializer(1.0)))
            self.vpred = tf.layers.dense(x, 1, name='value', kernel_initializer=U.normc_initializer(1.0))
            self.vpredz = self.vpred

        stochastic = tf.placeholder(dtype=tf.bool, shape=())
        ac = self.pd.sample()
        self._act = U.function([stochastic, ob], [ac, self.vpred])

    def act(self, stochastic, ob):
        _obs = {}
        for key, value in ob.items():
            _obs[key] = np.array(ob[key])[None]
        ac1, vpred1 = self._act(stochastic, _obs)
        return ac1[0], vpred1[0]
    def get_variables(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.scope)
    def get_trainable_variables(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)
    def get_initial_state(self):
        return []
    def save(self, session, path):
        ps = session.run(self.get_variables())
        joblib.dump(ps, path)
    def load(self, session, path):
        loaded_params = joblib.load(path)
        restores = []
        for p, loaded_p in zip(self.get_variables(), loaded_params):
            restores.append(p.assign(loaded_p))
        session.run(restores)
