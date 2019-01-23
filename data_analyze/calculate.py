import pickle
import numpy as np

log_dir = '/root/projects/robot_learning/experiments/trpo_top7/mean.txt'

with open(log_dir, 'rb') as file:
    b = pickle.load(file)
print(b)
mean_dic = {}
for k,v in b.items():
    mean_dic[k] = np.mean(v)
print(mean_dic)
