# -*- coding: utf-8 -*-
import os
import numpy as np

env_id = "ReacherBenchmarkEnv-v1"
# env_type = 0    # env类型：0=2轴, 1=6轴
LOG_DIR = os.path.join('experiments', 'test')
# LOG_DIR = os.path.join('Data', 'boxplot')

def Run_test(file_dir):

    print("current dir : {0}".format(file_dir))
    dir_list = os.listdir(file_dir)
    print('dir_list = ',dir_list)
    for cur_file in dir_list:
        if cur_file == '20190119-1526':
            pass
        else:
            print("cur_file=", cur_file)
            run_test = "python3 rllib/test_ppo1.py --env-id='"+env_id+"' --exp-id='"+cur_file+"'"
            print(run_test)
            os.system(run_test)


if __name__ == '__main__':

    file_dir11 = os.path.join(LOG_DIR, env_id)

    Run_test(file_dir11)