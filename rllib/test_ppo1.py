import gym
import time
import yaml
import numpy as np
import argparse
import os
import pprint
import csv
import pickle
import os.path as osp
import common.tf_util as U
from datetime import datetime

from common.tf_util import make_session
from common.misc_util import set_global_seeds
from envs import wrappers
from algorithms.ppo1 import cnn_policy, mlp_policy
from envs.reacher_benchmark_env.reacher_benchmark import clip_j, JOINT_ANGLE_LIMIT, ENV_CONFIG_PATH

# DEFAULT_LOG_DIR = os.path.join('Data', 'boxplot')  # 训练模型所在路径
DEFAULT_LOG_DIR = os.path.join('experiments', 'ppo1_R6')  # 训练模型所在路径
meanfile_name = "Mean_new.csv"
test_logName = "test.csv"
pickle_name = "test.txt"
test_config_dir = 'config/trpo_test_config.yaml'
env_config_dir = 'config/reacher_env_config.yaml'
obs_clip = clip_j    # 是否clip obs


def make_csv(filename, EXT):
    # 在filename文件夹下创建csv文件
    if not filename.endswith(EXT):
        if osp.isdir(filename):
            filename = osp.join(filename, EXT)
        else:
            filename = filename + "." + EXT
    f = open(filename, "at")
    return f


def GoThrough_XYZ_target():

    x_range = np.linspace(0, 30, 3)
    y_range = [0]
    z_range = np.linspace(0, 30, 3)
    all_pos = []
    for i in range(len(x_range)):
        for j in range(len(z_range)):
            for k in range(len(y_range)):
                all_pos.append([x_range[i]+300, y_range[k], z_range[j]+200])

    print("max_episodes:", np.shape(all_pos)[0])
    return all_pos


def CalculateMean(args, config, total_rew, total_dis, log_dir):
    # 创建Mean.csv文件(记录每个实验的平均测试结果, 保存在总文件夹下)
    filename_top = log_dir
    filename_mean1 = make_csv(DEFAULT_LOG_DIR, meanfile_name)
    logger_mean1 = csv.DictWriter(filename_mean1, fieldnames=('exp_id', 'timesteps_per_actorbatch', 'clip_param',
                                                            'optim_epochs', 'optim_stepsize', 'optim_batchsize',
                                                            'hidden_size', 'hidden_layers', 'MeanRew', 'MeanDis'))
    logger_mean1.writeheader()
    filename_mean1.flush()
    meaninfo = {"exp_id": args.exp_id, "timesteps_per_actorbatch": config['timesteps_per_actorbatch'],"clip_param": config['clip_param'],
                "optim_epochs": config['optim_epochs'], "optim_stepsize": config['optim_stepsize'], "optim_batchsize": config['optim_batchsize'],
                "hidden_size": config['policy']['hidden_size'], "hidden_layers": config['policy']['hidden_layers'],
                "MeanRew": np.mean(total_rew), "MeanDis": np.mean(total_dis)}
    if logger_mean1:
        logger_mean1.writerow(meaninfo)
        filename_mean1.flush()

    filename_mean = make_csv(filename_top, meanfile_name)
    logger_mean = csv.DictWriter(filename_mean, fieldnames=('exp_id', 'timesteps_per_actorbatch', 'clip_param',
                                                            'optim_epochs', 'optim_stepsize', 'optim_batchsize',
                                                            'hidden_size', 'hidden_layers', 'MeanRew', 'MeanDis'))
    logger_mean.writeheader()
    filename_mean.flush()
    picinfo = {"exp_id": [], "timesteps_per_actorbatch": [], "clip_param": [],
                "optim_epochs": [], "optim_stepsize": [], "optim_batchsize": [],
                "hidden_size": [], "hidden_layers": [], "MeanRew": [], "MeanDis": []}
    if not os.path.exists(log_dir+'/mean.txt'):
        # 创建Mean.txt文件(pickle)
        with open(log_dir+'/mean.txt', 'wb') as file:
            pickle.dump(picinfo, file)
    # 写入Mean.csv
    meaninfo = {"exp_id": args.exp_id, "timesteps_per_actorbatch": config['timesteps_per_actorbatch'],"clip_param": config['clip_param'],
                "optim_epochs": config['optim_epochs'], "optim_stepsize": config['optim_stepsize'], "optim_batchsize": config['optim_batchsize'],
                "hidden_size": config['policy']['hidden_size'], "hidden_layers": config['policy']['hidden_layers'],
                "MeanRew": np.mean(total_rew), "MeanDis": np.mean(total_dis)}
    if logger_mean:
        logger_mean.writerow(meaninfo)
        filename_mean.flush()

    with open(log_dir + '/mean.txt', 'rb') as file:
        beforeMean = pickle.load(file)
    for key in beforeMean.keys():
        beforeMean[key].append(meaninfo[key])
    with open(log_dir + '/mean.txt', 'wb') as file:
        pickle.dump(beforeMean, file)


def make_test_file(log_dir):
    # 创建test.csv文件(记录每个episode的测试结果,保存在对应的测试文件夹下)
    filename = log_dir
    f = make_csv(filename, test_logName)
    # f.write('#%s\n' % json.dumps({'env_id': env.spec and env.spec.id,
    #                               'vf_stepsize': config['vf_stepsize'],
    #                               'hidden_size': config['policy']['hidden_size'],
    #                               'hidden_layers': config['policy']['hidden_layers'],
    #                               'vf_iters': config['vf_iters'],
    #                               'timesteps_per_batch': config['timesteps_per_batch']}))

    logger = csv.DictWriter(f, fieldnames=('episode_rew', 'distance', 'target_pos'))
    logger.writeheader()
    f.flush()
    return logger, f


def write_test_csv(logger, f, epinfo):
    logger.writerow(epinfo)
    f.flush()


def write_pickle_file(log_dir,file_name, picinfo):
    with open(log_dir + '/' + file_name, 'wb') as file:
        pickle.dump(picinfo, file)


def Compute_var(obs, action, compute_var):
    obs = np.array(obs).reshape(1, 9)
    action = np.array(action).reshape(1, 6)
    args1 = obs, action
    ob_var = compute_var(*args1)
    return ob_var


def main():
    assert (JOINT_ANGLE_LIMIT == 5), "JOINT_ANGLE_LIMIT in reacher_benchmark.py must be 5!"
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--seed', help='random seed', default=8)
    parser.add_argument('--exp-id', help='experiment id', default=None)
    parser.add_argument('--env-id', help='environment id', default='ReacherBenchmarkEnv-v1')

    args = parser.parse_args()
    if args.env_id.startswith('ReacherBenchmarkGazeboEnv'):
        env_id_name = 'ReacherBenchmarkEnv-v1'
    else:
        env_id_name = args.env_id
    log_dir = os.path.join(DEFAULT_LOG_DIR, env_id_name, args.exp_id)
    # log_dir = os.path.join(DEFAULT_LOG_DIR, args.exp_id)
    train_env_config_dir = os.path.join(log_dir, 'ENV_CONFIG')

    # read train_env_config file
    with open(train_env_config_dir, 'r') as train_env_config_file:
        train_env_config = yaml.load(train_env_config_file)
    # read test_env_config file
    with open(ENV_CONFIG_PATH , 'r') as env_config_file:
        test_env_config = yaml.load(env_config_file)

    # read test_config file
    with open(test_config_dir, 'r') as test_config_file:
        test_config = yaml.load(test_config_file)
    # make test dir
    test_dir = os.path.join(log_dir, 'test0', test_config['test_type'], test_config['ObjTestType'],
                            datetime.now().strftime('%Y%m%d-%H%M'))
    if not osp.isdir(test_dir):
        os.makedirs(test_dir, exist_ok=True)

    # read train_config file
    config = yaml.load(open(os.path.join(log_dir, 'CONFIG')))
    pprint.pprint(config)

    # write test_config file
    testType = test_config['test_type']
    obj_pos = test_config['given_obj_pos']
    test_info = {'test_type': test_config['test_type'],
                 'ObjTestType': test_config['ObjTestType'],
                 'max_episodes': test_config['max_episodes'],
                 'given_obj_pos': test_config['given_obj_pos'],
                 'all_pos': test_config['all_pos'],
                 'train_env_config': train_env_config,
                 'test_env_config': test_env_config, }
    yaml.dump(test_info, open(os.path.join(test_dir, 'TEST_CONFIG'), 'w'))

    # make env
    set_global_seeds(int(args.seed))
    if args.env_id.startswith('PickPlace'):
        env = gym.make(args.env_id)
        env = wrappers.wrap_pick_place(env, obs_type=config['obs_type'], reward_type=config['reward_type'])
    elif args.env_id.startswith('Reacher'):
        env = gym.make(args.env_id)
        env = wrappers.wrap_reacher(env)
    elif args.env_id.startswith('CarRacing'):
        env = gym.make(args.env_id)
    else:
        env = wrappers.make_atari(args.env_id)
        env = wrappers.wrap_deepmind(env, frame_stack=True)
    # env.render(mode='human')
    # time.sleep(5)
    policy_config = config['policy']
    if config['obs_type'] == 'image_only':
        policy = cnn_policy.CnnPolicy(name='pi', ob_space=env.observation_space, ac_space=env.action_space, kind='small')
    elif config['obs_type'] == 'image_with_pose':
        policy = cnn_policy.CnnPolicyDict(name='pi', ob_space=env.observation_space, ac_space=env.action_space)
    elif config['obs_type'] == 'pose_only':
        policy = mlp_policy.MlpPolicy(name='pi', ob_space=env.observation_space, ac_space=env.action_space,
                              hid_size=policy_config['hidden_size'],
                              num_hid_layers=policy_config['hidden_layers'],
                              gaussian_fixed_var=policy_config['gaussian_fixed_var'])

    # load model
    sess = make_session(make_default=True)
    model_dir = os.path.join(log_dir, 'model')
    # print('******3')

    policy.load(sess, model_dir)
    # print('******4')


    # make csv file
    if test_config[testType]['save_csv']:
        logger, f = make_test_file(test_dir)

    var = policy.pd.std
    ob = U.get_placeholder_cached(name="ob")
    ac = policy.pdtype.sample_placeholder([None])
    # atarg = tf.placeholder(dtype=tf.float32, shape=[None]) # Target advantage function (if applicable)
    compute_var = U.function([ob, ac], var)

    total_rew = []
    total_dis = []
    total_pos = []

    # define max_episodes and given_pos for ObjTestType
    assert test_config['ObjTestType'] in ['Random', 'GoThrough_XYZ', 'AllPos', 'GivenPos']
    if 'GivenPos' == test_config['ObjTestType']:
        max_episodes = 1
        given_pos = obj_pos
        print("Test one given obj pos!")
    elif 'Random' == test_config['ObjTestType']:
        max_episodes = test_config['max_episodes']
        given_pos = [None] * max_episodes
        print("Random test! // Test number:", max_episodes)
    elif 'GoThrough_XYZ' == test_config['ObjTestType']:
        given_pos = GoThrough_XYZ_target()
        max_episodes = np.shape(given_pos)[0]
        print("GoThrough_XYZ test! // Test number:", max_episodes)
    elif 'AllPos' == test_config['ObjTestType']:
        given_pos = test_config['all_pos']
        max_episodes = np.shape(given_pos)[0]
        print("All Given Pos test! // Test number:", max_episodes)


    # make plot_dir
    plot_dir = os.path.join(test_dir, 'plot')
    if not osp.isdir(plot_dir):
        os.makedirs(plot_dir, exist_ok=True)

    # start test
    for i in range(max_episodes):

        obs = env.reset(given_pos[i])
        done = False
        episode_rew = 0
        total_action = []
        total_obs = []
        total_var = []
        while not done:
            action = policy.act(0, obs)[0]  # 0 stochastic = False
            ob_var = Compute_var(obs, action, compute_var)
            total_action.append(action)
            total_obs.append(obs)
            total_var.append(ob_var)
            obs, rew, done, info = env.step(action)
            episode_rew += rew

        dis = np.sqrt(np.square(obs[-3])+np.square(obs[-2])+np.square(obs[-1]))  # 计算distance
        print("Episode reward=", episode_rew)
        print("Distance=", dis)
        total_rew.append(episode_rew)
        total_dis.append(dis)
        total_pos.append(info['object_pos'])

        # define plot_name by object_pos info
        plot_name = str(int(info['object_pos'][0])) + '_' + \
                    str(int(info['object_pos'][1])) + '_' + \
                    str(int(info['object_pos'][2]))

        # 写入test.csv
        csvinfo = {"episode_rew": round(episode_rew, 6), "distance": round(dis, 6), "target_pos": info['object_pos']}
        if test_config[testType]['save_csv']:
            write_test_csv(logger, f, csvinfo)

        # 保存1个episode的obs
        if test_config[testType]['save_obs']:
            if obs_clip:
                write_pickle_file(plot_dir, plot_name + "obs_clip.txt", total_obs)
                write_pickle_file(plot_dir, plot_name + "distance.txt", dis)
                if test_config[testType]['save_action']:
                    write_pickle_file(plot_dir, plot_name + "action.txt", total_action)
            else:
                write_pickle_file(plot_dir, plot_name + "obs_no_clip.txt", total_obs)

        # 保存1个episode的var
        if test_config[testType]['save_var']:
            varinfo = {"action": total_action, "var": total_var}
            write_pickle_file(plot_dir, plot_name + "var.txt", varinfo)

    # 保存所有episode的reward, distance, target_pos
    picinfo = {"episode_rew": total_rew, "distance": total_dis, "target_pos": total_pos}
    if test_config[testType]['save_pickle']:
        write_pickle_file(test_dir, pickle_name, picinfo)

    # 保存所有episode下的平均reward, 平均distance
    if test_config[testType]['save_mean']:
        mean_dir = os.path.join(log_dir, 'test0')
        CalculateMean(args, config, total_rew, total_dis, mean_dir)

    env.close()


if __name__ == '__main__':
    main()
    # GoThrough_XYZ_target()
    # env = gym.make('Reacher2BenchmarkEnv-v0')
    # import sys
    # sys.path.append('..')
    # from rllib.envs.reacher_benchmark_env.reacher_benchmark import ReacherBenchmarkEnv
    #
    # env = ReacherBenchmarkEnv('uniform', [1, 2], 'PIROBOT')
    # env.get_reset_obj_pos([1,2,3])
    # obs = env.reset()
    # print(obs)