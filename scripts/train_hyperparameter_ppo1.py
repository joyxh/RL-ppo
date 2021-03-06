from itertools import product
import yaml
import os

# seed = 0
env_id = "'ReacherBenchmarkEnv-v1'"
config_file_path = os.path.join('config', 'ppo1_reacher_traverse.yaml')
new_config_file = os.path.join('config', 'ppo1_default.yaml')   # 输入新建配置文件路径
train_file = os.path.join('rllib', 'train_ppo1_mpi.py')   # 输入需要运行的训练代码所在路径
osInput = "mpirun -n 4 --allow-run-as-root  python3 "+train_file+" --env-id="+env_id+" --config-file="+new_config_file

def flatten_dict(d):
    """拉平字典，示例：输入 d={'a': {'b':'c'}}，输出{'a.b': 'c'}"""
    _d = {}
    for k1, v1 in d.items():
        if not isinstance(v1, dict):
            _d[k1] = v1
        else:
            flat_d = flatten_dict(v1)
            for k2, v2 in flat_d.items():
                _d['.'.join([k1, k2])] = v2
    return _d


def rebuild_dict(d, keys, values):

    def set_nested_dict(d, keys, value):
        if len(keys) > 1:
            d[keys[0]] = set_nested_dict(d[keys[0]], keys[1:], value)
        else:
            d[keys[0]] = value
        return d

    keys = list(keys)
    for i in range(len(keys)):
        ks = keys[i].split('.')
        set_nested_dict(d, ks, values[i])
    return d

def set_nested_dict(d, keys, value):
    if len(keys) > 1:
        d[keys[0]] = set_nested_dict(d[keys[0]], keys[1:], value)
    else:
        d[keys[0]] = value
    return d


def nested_hyperparams_generator(d):

    flat_dict = flatten_dict(d)
    for k, v in flat_dict.items():
        if not isinstance(v, list):
            flat_dict[k] = [v]

    keys = list(flat_dict.keys())
    values = flat_dict.values()

    param_combinations = product(*values)
    for comb in param_combinations:
        yield rebuild_dict(d, keys, comb)


def skip_hyperparams(c, g):
    timesteps_per_actorbatch = c['timesteps_per_actorbatch']
    optim_batchsize = c['optim_batchsize']
    if timesteps_per_actorbatch < optim_batchsize:
        print('timesteps_per_actorbatch is less than optim_batchsize, stop training!')
        c = next(g)
        skip_hyperparams(c, g)


def main():
    config = yaml.load(open(config_file_path))

    g = nested_hyperparams_generator(config)

    i = 0
    for c in g:
        print(c)
        skip_hyperparams(c, g)
        with open(new_config_file,'w') as yaml_file:
            yaml.dump(c, yaml_file)
        # yaml.dump(c, open(os.path.join('experiments', 'new_test', 'test' + str(i)), 'w'))
        # i += 1
        os.system(osInput)


if __name__ == '__main__':
    main()
