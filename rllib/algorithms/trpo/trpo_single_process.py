from common.math_util import explained_variance
from common.misc_util import zipsame
from common import dataset
from common import logger
import common.tf_util as U
import tensorflow as tf
import numpy as np
import time
from common.console_util import colorize
from collections import deque
from common.cg import cg
from contextlib import contextmanager
import os
import gym
import pickle


def traj_segment_generator(pi, env, horizon, stochastic):
    # Initialize state variables
    t = 0
    ac = env.action_space.sample()
    new = True
    rew = 0.0
    ob = env.reset()

    cur_ep_ret = 0
    cur_ep_len = 0
    ep_rets = []
    ep_lens = []

    # Initialize history arrays
    if isinstance(ob, dict):
        obs = [ob for _ in range(horizon)]
    else:
        obs = np.array([ob for _ in range(horizon)])
    rews = np.zeros(horizon, 'float32')
    vpreds = np.zeros(horizon, 'float32')
    news = np.zeros(horizon, 'int32')
    acs = np.array([ac for _ in range(horizon)])
    prevacs = acs.copy()

    while True:
        prevac = ac
        ac, vpred = pi.act(stochastic, ob)
        # Slight weirdness here because we need value function at time T
        # before returning segment [0, T-1] so we get the correct
        # terminal value
        if t > 0 and t % horizon == 0:
            yield {"ob" : obs, "rew" : rews, "vpred" : vpreds, "new" : news,
                    "ac" : acs, "prevac" : prevacs, "nextvpred": vpred * (1 - new),
                    "ep_rets" : ep_rets, "ep_lens" : ep_lens}
            _, vpred = pi.act(stochastic, ob)
            # Be careful!!! if you change the downstream algorithm to aggregate
            # several of these batches, then be sure to do a deepcopy
            ep_rets = []
            ep_lens = []
        i = t % horizon
        obs[i] = ob
        vpreds[i] = vpred
        news[i] = new
        acs[i] = ac
        prevacs[i] = prevac

        ob, rew, new, _ = env.step(ac)
        rews[i] = rew

        cur_ep_ret += rew
        cur_ep_len += 1
        if new:
            ep_rets.append(cur_ep_ret)
            ep_lens.append(cur_ep_len)
            # dis = np.sqrt(np.square(ob[-3])+np.square(ob[-1]))
            # logger.record_tabular("Distance", dis)
            # # print("ob=====",ob[-3:])
            cur_ep_ret = 0
            cur_ep_len = 0
            ob = env.reset()
        t += 1

def add_vtarg_and_adv(seg, gamma, lam):
    new = np.append(seg["new"], 0) # last element is only used for last vtarg, but we already zeroed it if last new = 1
    vpred = np.append(seg["vpred"], seg["nextvpred"])
    T = len(seg["rew"])
    seg["adv"] = gaelam = np.empty(T, 'float32')
    rew = seg["rew"]
    lastgaelam = 0
    for t in reversed(range(T)):
        nonterminal = 1-new[t+1]
        delta = rew[t] + gamma * vpred[t+1] * nonterminal - vpred[t]
        gaelam[t] = lastgaelam = delta + gamma * lam * nonterminal * lastgaelam
    seg["tdlamret"] = seg["adv"] + seg["vpred"]

def learn(env, policy_fn, *,
        timesteps_per_batch, # what to train on
        max_kl, cg_iters,
        gamma, lam, # advantage estimation
        entcoeff=0.0,
        cg_damping=1e-2,
        vf_stepsize=3e-4,
        vf_iters =3,
        max_timesteps=0, max_episodes=0, max_iters=0,  # time constraint
        callback=None
        ):
    np.set_printoptions(precision=3)
    # Setup losses and stuff
    # ----------------------------------------
    if isinstance(env.observation_space, gym.spaces.Dict):
        ob_space = env.observation_space.spaces
    else:
        ob_space = env.observation_space
    ac_space = env.action_space

    pi = policy_fn("pi", ob_space, ac_space)
    oldpi = policy_fn("oldpi", ob_space, ac_space)
    atarg = tf.placeholder(dtype=tf.float32, shape=[None]) # Target advantage function (if applicable)
    ret = tf.placeholder(dtype=tf.float32, shape=[None]) # Empirical return

    if isinstance(env.observation_space, gym.spaces.Dict):
        from algorithms.deepq.utils import DictTfInput
        image = U.get_placeholder_cached(name='image')
        robot_pos = U.get_placeholder_cached(name='robot_pos')
        ob = DictTfInput(None, ob_space, {
            'image': image,
            'robot_pos': robot_pos
        })
    else:
        ob = U.get_placeholder_cached(name="ob")

    ac = pi.pdtype.sample_placeholder([None])

    kloldnew = oldpi.pd.kl(pi.pd)
    ent = pi.pd.entropy()
    meankl = tf.reduce_mean(kloldnew)
    meanent = tf.reduce_mean(ent)
    entbonus = entcoeff * meanent
    var = pi.pd.std

    vferr = tf.reduce_mean(tf.square(pi.vpred - ret))
    # vferr = tf.Print(vferr, [pi.vpred, ret])

    ratio = tf.exp(pi.pd.logp(ac) - oldpi.pd.logp(ac)) # advantage * pnew / pold
    surrgain = tf.reduce_mean(ratio * atarg)

    optimgain = surrgain + entbonus
    losses = [optimgain, meankl, entbonus, surrgain, meanent]
    loss_names = ["optimgain", "meankl", "entloss", "surrgain", "entropy"]

    dist = meankl

    all_var_list = pi.get_trainable_variables()
    var_list = [v for v in all_var_list if v.name.split("/")[1].startswith("pol")]
    vf_var_list = [v for v in all_var_list if v.name.split("/")[1].startswith("vf")]

    assign_old_eq_new = U.function([], [], updates=[tf.assign(oldv, newv) for (oldv, newv) in zipsame(
        oldpi.get_variables(), pi.get_variables())])

    vftrainer = tf.train.RMSPropOptimizer(learning_rate=vf_stepsize)
    vf_train = vftrainer.minimize(vferr, var_list=vf_var_list)

    get_flat = U.GetFlat(var_list)
    set_from_flat = U.SetFromFlat(var_list)
    klgrads = tf.gradients(dist, var_list)
    flat_tangent = tf.placeholder(dtype=tf.float32, shape=[None], name="flat_tan")
    shapes = [var.get_shape().as_list() for var in var_list]
    start = 0
    tangents = []
    for shape in shapes:
        sz = U.intprod(shape)
        tangents.append(tf.reshape(flat_tangent[start:start+sz], shape))
        start += sz
    gvp = tf.add_n([tf.reduce_sum(g*tangent) for (g, tangent) in zipsame(klgrads, tangents)]) #pylint: disable=E1111
    fvp = U.flatgrad(gvp, var_list)

    compute_losses = U.function([ob, ac, atarg], losses)
    compute_lossandgrad = U.function([ob, ac, atarg], losses + [U.flatgrad(optimgain, var_list)])
    compute_fvp = U.function([flat_tangent, ob, ac, atarg], fvp)
    compute_vflossandgrad = U.function([ob, ret], U.flatgrad(vferr, vf_var_list))
    vfadam_update = U.function([ob, ret], outputs=[vferr], updates=[vf_train])
    # compute_var = U.function([ob, ac, atarg], var)

    @contextmanager
    def timed(msg):
        print(colorize(msg, color='magenta'))
        tstart = time.time()
        yield
        print(colorize("done in %.3f seconds"%(time.time() - tstart), color='magenta'))
        logger.record_tabular('time_%s' % msg, time.time() - tstart)

    assert sum([max_iters > 0, max_timesteps > 0, max_episodes > 0]) == 1

    sess = U.make_session(make_default=True)

    U.initialize()

    # Prepare for rollouts
    # ----------------------------------------
    seg_gen = traj_segment_generator(pi, env, timesteps_per_batch, stochastic=True)

    checkpoint_dir = os.path.join(logger.get_dir(), 'checkpoint')
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, 'model')
    model_path = os.path.join(logger.get_dir(), 'model')

    loop_params = {
        'episodes_so_far': 0,
        'timesteps_so_far': 0,
        'iters_so_far': 0,
        'lenbuffer': deque(maxlen=40),  # rolling buffer for episode lengths
        'rewbuffer': deque(maxlen=40),  # rolling buffer for episode rewards
        'best_mean_reward': None
    }

    loop_params_file = os.path.join(checkpoint_dir, 'loop_params.pkl')
    if os.path.exists(loop_params_file):
        logger.info('Loading loop parameters from', loop_params_file)
        loop_params.update((pickle.load(open(loop_params_file, 'rb'))))

    episodes_so_far = loop_params['episodes_so_far']
    timesteps_so_far = loop_params['timesteps_so_far']
    iters_so_far = loop_params['iters_so_far']
    lenbuffer = loop_params['lenbuffer']
    rewbuffer = loop_params['rewbuffer']
    best_mean_reward = loop_params['best_mean_reward']
    # tstart = time.time()

    saver = tf.train.Saver()
    if tf.train.latest_checkpoint(checkpoint_dir) is not None:
        logger.info('Loading model from {}'.format(model_path))
        saver.restore(tf.get_default_session(), model_path)

    while True:
        if callback: callback(locals(), globals())
        if max_timesteps and timesteps_so_far >= max_timesteps:
            break
        elif max_episodes and episodes_so_far >= max_episodes:
            break
        elif max_iters and iters_so_far >= max_iters:
            break
        logger.log("********** Iteration %i ************"%iters_so_far)

        with timed("sampling"):
            seg = seg_gen.__next__()
        add_vtarg_and_adv(seg, gamma, lam)
        # input()

        # ob, ac, atarg, ret, td1ret = map(np.concatenate, (obs, acs, atargs, rets, td1rets))
        ob, ac, atarg, tdlamret = seg["ob"], seg["ac"], seg["adv"], seg["tdlamret"]
        # logger.log(" obs:", ob)

        vpredbefore = seg["vpred"] # predicted value function before udpate
        atarg = (atarg - atarg.mean()) / atarg.std() # standardized advantage function estimate

        if hasattr(pi, "ret_rms"): pi.ret_rms.update(tdlamret)
        if hasattr(pi, "ob_rms"): pi.ob_rms.update(ob) # update running mean/std for policy

        args = seg["ob"], seg["ac"], atarg

        # ob_var = compute_var(*args)
        # logger.log("var:", ob_var)

        fvpargs = [arr[::5] for arr in args]
        def fisher_vector_product(p):
            return compute_fvp(p, *fvpargs) + cg_damping * p

        assign_old_eq_new() # set old parameter values to new parameter values
        with timed("computegrad"):
            *lossbefore, g = compute_lossandgrad(*args)
        # lossbefore = np.array(lossbefore)
        if np.allclose(g, 0):
            logger.log("Got zero gradient. not updating")
        else:
            with timed("cg"):
                stepdir = cg(fisher_vector_product, g, cg_iters=cg_iters, verbose=True)
            assert np.isfinite(stepdir).all()
            shs = .5*stepdir.dot(fisher_vector_product(stepdir))
            lm = np.sqrt(shs / max_kl)
            logger.log("lagrange multiplier:", lm, " gnorm:", np.linalg.norm(g))
            fullstep = stepdir / lm
            expectedimprove = g.dot(fullstep)
            surrbefore = lossbefore[0]
            stepsize = 1.0
            thbefore = get_flat()
            for _ in range(10):
                thnew = thbefore + fullstep * stepsize
                set_from_flat(thnew)
                meanlosses = surr, kl, *_ = np.array(compute_losses(*args))
                improve = surr - surrbefore
                logger.log("Expected: %.3f Actual: %.3f"%(expectedimprove, improve))
                logger.log("KL: %.5f, max kl * 1.5: %.5f" % (kl, max_kl*1.5))
                if not np.isfinite(meanlosses).all():
                    logger.log("Got non-finite value of losses -- bad!")
                elif kl > max_kl * 1.5:
                    logger.log("violated KL constraint. shrinking step.")
                elif improve < 0:
                    logger.log("surrogate didn't improve. shrinking step.")
                else:
                    logger.log("Stepsize OK!")
                    break
                stepsize *= .5
            else:
                logger.log("couldn't compute a good step")
                set_from_flat(thbefore)

        for (lossname, lossval) in zip(loss_names, meanlosses):
            logger.record_tabular(lossname, lossval)

        with timed("vf"):
            for i in range(vf_iters):
                for (mbob, mbret) in dataset.iterbatches((seg["ob"], seg["tdlamret"]),
                include_final_partial_batch=False, batch_size=64):
                    # print(mbob, mbret)
                    vf_loss = vfadam_update(mbob, mbret)[0]
                logger.log("value function loss %d: %.5f" % (i, vf_loss))

        logger.record_tabular("ev_tdlam_before", explained_variance(vpredbefore, tdlamret))

        lrlocal = (seg["ep_lens"], seg["ep_rets"]) # local values
        lens, rews = lrlocal
        lenbuffer.extend(lens)
        rewbuffer.extend(rews)

        logger.record_tabular("EpLenMean", np.mean(lenbuffer))
        mean_reward = np.mean(rewbuffer)
        logger.record_tabular("EpRewMean", mean_reward)
        logger.record_tabular("EpThisIter", len(lens))
        episodes_so_far += len(lens)
        timesteps_so_far += sum(lens)
        iters_so_far += 1

        logger.record_tabular("EpisodesSoFar", episodes_so_far)
        logger.record_tabular("TimestepsSoFar", timesteps_so_far)
        # logger.record_tabular("TimeElapsed", time.time() - tstart)

        logger.dump_tabular()

        # checkpoint
        logger.info('Dumping checkpoints...')
        saver.save(tf.get_default_session(), checkpoint_path)
        loop_params.update({
            'episodes_so_far': episodes_so_far,
            'timesteps_so_far': timesteps_so_far,
            'iters_so_far': iters_so_far,
            'lenbuffer': lenbuffer,  # rolling buffer for episode lengths
            'rewbuffer': rewbuffer,  # rolling buffer for episode rewards
            'best_mean_reward': best_mean_reward
        })
        pickle.dump(loop_params, open(loop_params_file, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)

        # save best model
        if best_mean_reward is None or mean_reward > best_mean_reward:
            logger.log("Saving model due to mean reward increase: {} -> {}".format(
                best_mean_reward, mean_reward))
            pi.save(sess, model_path)
            best_mean_reward = mean_reward


def flatten_lists(listoflists):
    return [el for list_ in listoflists for el in list_]