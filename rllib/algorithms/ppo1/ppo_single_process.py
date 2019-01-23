from common.dataset import Dataset
from common.math_util import explained_variance
from common.misc_util import zipsame
from common.console_util import fmt_row
from common import logger
import common.tf_util as U
import tensorflow as tf, numpy as np
from common.console_util import colorize
from collections import deque
from contextlib import contextmanager
import os
import time
import gym
import pickle

def traj_segment_generator(pi, env, horizon, stochastic):
    t = 0
    ac = env.action_space.sample() # not used, just so we have the datatype
    new = True # marks if we're on first timestep of an episode
    ob = env.reset()

    cur_ep_ret = 0 # return in current episode
    cur_ep_len = 0 # len of current episode
    ep_rets = [] # returns of completed episodes in this segment
    ep_lens = [] # lengths of ...

    # Initialize history arrays
    if isinstance(ob, dict):
        obs = [ob for _ in range(horizon)]
    else:
        obs = np.array([ob for _ in range(horizon)])
    # obs = np.array([ob for _ in range(horizon)])
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
            cur_ep_ret = 0
            cur_ep_len = 0
            ob = env.reset()
        t += 1

def add_vtarg_and_adv(seg, gamma, lam):
    """
    Compute target value using TD(lambda) estimator, and advantage with GAE(lambda)
    """
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
    print('************gaelam[t],lastgaelam = ', gaelam[T-1],lastgaelam)
    print('****meanreward,vpred[t+1],vpred[t] = ', np.mean(np.array(rew)), vpred[T], vpred[T-1])
    print('*****nonterminal = ', nonterminal)
    print('*******delta = ', delta)
    seg["tdlamret"] = seg["adv"] + seg["vpred"]

def learn(env, policy_fn, *,
        timesteps_per_actorbatch, # timesteps per actor per update
        clip_param, entcoeff, # clipping parameter epsilon, entropy coeff
        optim_epochs, optim_stepsize, optim_batchsize,# optimization hypers
        gamma, lam, # advantage estimation
        max_timesteps=0, max_episodes=0, max_iters=0, max_seconds=0,  # time constraint
        callback=None, # you can do anything in the callback, since it takes locals(), globals()
        adam_epsilon=1e-5,
        schedule='constant' # annealing for stepsize parameters (epsilon and adam)
        ):
    # Setup losses and stuff
    # ----------------------------------------
    ob_space = env.observation_space
    ac_space = env.action_space

    pi = policy_fn("pi", ob_space, ac_space) # Construct network for new policy
    oldpi = policy_fn("oldpi", ob_space, ac_space) # Network for old policy
    atarg = tf.placeholder(dtype=tf.float32, shape=[None]) # Target advantage function (if applicable)
    ret = tf.placeholder(dtype=tf.float32, shape=[None]) # Empirical return

    lrmult = tf.placeholder(name='lrmult', dtype=tf.float32, shape=[]) # learning rate multiplier, updated with schedule
    clip_param = clip_param * lrmult # Annealed cliping parameter epislon

    ob = U.get_placeholder_cached(name="ob")
    ac = pi.pdtype.sample_placeholder([None])

    kloldnew = oldpi.pd.kl(pi.pd)
    ent = pi.pd.entropy()
    meankl = tf.reduce_mean(kloldnew)
    meanent = tf.reduce_mean(ent)
    pol_entpen = (-entcoeff) * meanent

    ratio = tf.exp(pi.pd.logp(ac) - oldpi.pd.logp(ac)) # pnew / pold
    # ratio = tf.Print(ratio, [ratio], message='*********')
    surr1 = ratio * atarg # surrogate from conservative policy iteration
    q = tf.clip_by_value(ratio, 1.0 - clip_param, 1.0 + clip_param)
    surr2 = q * atarg #
    # su1 = tf.Print(surr1[:5], [surr1[:5]], message='########')
    # su2 = tf.Print(surr2[:5], [surr2[:5]], message='########')
    w = tf.reduce_mean(q)
    # q = tf.Print(q, [q], message='**************')
    # print('*********q = ',q)
    s1 = tf.reduce_mean(surr1)
    s2 = tf.reduce_mean(surr2)
    r = tf.reduce_mean(ratio)
    pol_surr = - tf.reduce_mean(tf.minimum(surr1, surr2)) # PPO's pessimistic surrogate (L^CLIP)
    vf_loss = tf.reduce_mean(tf.square(pi.vpred - ret))
    total_loss = pol_surr + pol_entpen + vf_loss
    losses = [pol_surr, pol_entpen, vf_loss, meankl, meanent]
    # loss_names = ["pol_surr", "pol_entpen", "vf_loss", "kl", "ent"]
    losses = [r, w, s1, s2, pol_surr, pol_entpen, vf_loss, meankl, meanent]
    loss_names = ["r", "w", "surr1", "surr2", "pol_surr", "pol_entpen", "vf_loss", "kl", "ent"]

    var_list = pi.get_trainable_variables()
    lossandgrad = U.function([ob, ac, atarg, ret, lrmult], losses + [U.flatgrad(total_loss, var_list)])
    # adam = MpiAdam(var_list, epsilon=adam_epsilon)
    assign_old_eq_new = U.function([], [], updates=[tf.assign(oldv, newv) for (oldv, newv) in zipsame(
        oldpi.get_variables(), pi.get_variables())])

    # vftrainer = tf.train.RMSPropOptimizer(learning_rate=lrmult)
    vftrainer = tf.train.AdamOptimizer(learning_rate=lrmult, epsilon=adam_epsilon)
    vf_train = vftrainer.minimize(total_loss, var_list=var_list)

    compute_losses = U.function([ob, ac, atarg, ret, lrmult], losses)
    vfadam_update = U.function([ob, ac, atarg, ret, lrmult], outputs=[total_loss], updates=[vf_train])

    sess = U.make_session(make_default=True)

    @contextmanager
    def timed(msg):
        print(colorize(msg, color='magenta'))
        tstart = time.time()
        yield
        print(colorize("done in %.3f seconds"%(time.time() - tstart), color='magenta'))
        logger.record_tabular('time_%s' % msg, time.time() - tstart)

    U.initialize()

    # Prepare for rollouts
    # ----------------------------------------
    seg_gen = traj_segment_generator(pi, env, timesteps_per_actorbatch, stochastic=True)

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
    tstart = time.time()

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
        elif max_seconds and time.time() - tstart >= max_seconds:
            break

        if schedule == 'constant':
            cur_lrmult = 1.0
        elif schedule == 'linear':
            cur_lrmult = max(1.0 - float(timesteps_so_far) / max_timesteps, 0)
        else:
            raise NotImplementedError

        logger.log("********** Iteration %i ************"%iters_so_far)

        with timed("sampling"):
            seg = seg_gen.__next__()
        add_vtarg_and_adv(seg, gamma, lam)

        # ob, ac, atarg, ret, td1ret = map(np.concatenate, (obs, acs, atargs, rets, td1rets))
        ob, ac, atarg, tdlamret = seg["ob"], seg["ac"], seg["adv"], seg["tdlamret"]
        print('******************atarg = ', atarg)
        vpredbefore = seg["vpred"] # predicted value function before udpate
        atarg = (atarg - atarg.mean()) / atarg.std() # standardized advantage function estimate
        d = Dataset(dict(ob=ob, ac=ac, atarg=atarg, vtarg=tdlamret), shuffle=not pi.recurrent)
        optim_batchsize = optim_batchsize or ob.shape[0]

        if hasattr(pi, "ob_rms"): pi.ob_rms.update(ob) # update running mean/std for policy

        assign_old_eq_new() # set old parameter values to new parameter values
        logger.log("Optimizing...")
        logger.log(fmt_row(13, loss_names))
        # Here we do a bunch of optimization epochs over the data
        with timed("optimizing"):
            for _ in range(optim_epochs):
                losses = [] # list of tuples, each of which gives the loss for a minibatch
                # meanlosses = []
                for batch in d.iterate_once(optim_batchsize):
                    newlosses = compute_losses(batch["ob"], batch["ac"], batch["atarg"], batch["vtarg"], cur_lrmult)
                    vfadam_update(batch["ob"], batch["ac"], batch["atarg"], batch["vtarg"], optim_stepsize * cur_lrmult)
                    # adam.update(g, optim_stepsize * cur_lrmult)
                    losses.append(newlosses)
                meanloss_ = np.mean(losses, axis=0)
                # meanlosses.append(meanloss_)
                logger.log(fmt_row(13, meanloss_))

        # logger.log("Evaluating losses...")
        # meanloss = np.mean(meanlosses, axis=0)
        # logger.log(fmt_row(13, meanloss))
        # for (lossval, name) in zipsame(meanloss, loss_names):
        #     logger.record_tabular("loss_"+name, lossval)

        logger.log("Evaluating losses...")
        with timed("Evaluating"):
            losses = []
            for batch in d.iterate_once(optim_batchsize):
                newlosses = compute_losses(batch["ob"], batch["ac"], batch["atarg"], batch["vtarg"], cur_lrmult)
                losses.append(newlosses)
                # logger.log('##newlosses##      ', newlosses)
            meanlosses = np.mean(losses, axis=0)
            logger.log(fmt_row(13, meanlosses))
        for (lossval, name) in zipsame(meanlosses, loss_names):
            logger.record_tabular("loss_"+name, lossval)
        logger.record_tabular("ev_tdlam_before", explained_variance(vpredbefore, tdlamret))
        lrlocal = (seg["ep_lens"], seg["ep_rets"]) # local values
        lens, rews = lrlocal
        lenbuffer.extend(lens)
        rewbuffer.extend(rews)
        logger.record_tabular("EpLenMean", np.mean(lenbuffer))
        mean_reward = np.mean(rewbuffer)
        logger.record_tabular("EpRewMean", np.mean(rewbuffer))
        logger.record_tabular("EpThisIter", len(lens))
        episodes_so_far += len(lens)
        timesteps_so_far += sum(lens)
        iters_so_far += 1
        logger.record_tabular("EpisodesSoFar", episodes_so_far)
        logger.record_tabular("TimestepsSoFar", timesteps_so_far)
        logger.record_tabular("TimeElapsed", time.time() - tstart)
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
