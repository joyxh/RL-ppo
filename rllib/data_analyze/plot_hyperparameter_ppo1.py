# -*- coding: utf-8 -*-
import os
import numpy as np

env_id = "ReacherBenchmarkEnv-v1"
# env_type = 0    # env类型：0=2轴, 1=6轴
LOG_DIR = 'experiments/ppo_trpo/ppo/random/'

def Run_plot(file_dir):
    print("current dir : {0}".format(file_dir))
    dir_list = os.listdir(file_dir)
    # print("dir_list = ", dir_list)
    for cur_file in dir_list:
        if cur_file == 'Mean_new.csv':
            pass
        else:
            print("cur_file=", cur_file)
            test_dir = os.path.join(file_dir, cur_file) + '/test0/TagetScatter/Random/'
            test_list = os.listdir(test_dir)
            for test_file in test_list:
                print("test_file=", test_file)
                run_plot = "python3 rllib/data_analyze/data_plot.py --env-id='" + env_id + "' --exp-id='" \
                           + cur_file + "' --test-id='" + test_file + "'"
                print(run_plot)
                os.system(run_plot)


if __name__ == '__main__':

    file_dir11 = LOG_DIR

    Run_plot(file_dir11)
# dir_list = ['20190101-2102', '20190101-1519', '20181230-2206', '20181231-0643', '20181231-1107', '20181231-1954',
#             '20190101-2120', '20181231-0101', '20190102-1331', '20181230-1910', '20190101-1738', '20181230-1434',
#             '20181231-1341', '20190102-0819', '20190102-0411', '20181230-2042', '20190101-0651', '20190101-0342',
#             '20181230-1613', '20181230-2339', '20181231-0520', '20181231-2355', '20181231-0351', '20181230-1747',
#             '20181231-1635', '20190101-1252', '20181231-0818', '20190101-1038', '20190102-0154', '20190101-2337',
#             '20181231-2045', '20181231-0228']
# a = []
# for cur_file in dir_list:
#     if cur_file == '20190102-1331':
#         pass
#     else:
#         a.append(cur_file)
# print(a)