import pickle
import numpy as np

log_dir = './Data/boxplot/20190109-1208/test/TagetScatter/Random/20190109-1606/test.txt'


def print_mean():
    with open(log_dir, 'rb') as file:
        b = pickle.load(file)
    print(b)
    mean_dic = {}
    for k,v in b.items():
        mean_dic[k] = np.mean(v)
    print(mean_dic)


def cal_accuracy(ranges):
    num = 0
    with open(log_dir, 'rb') as file:
        b = pickle.load(file)
    distance = b['distance']
    # print(distance[0])

    for i in range(len(distance)):
        if distance[i] < ranges:
            num += 1
    return num


if __name__ == '__main__':
    ac_10 = cal_accuracy(10)
    ac_20 = cal_accuracy(20)
    ac_30 = cal_accuracy(30)
    ac_100 = cal_accuracy(100)
    print("ac_10 = ", ac_10)
    print("ac_20 = ", ac_20)
    print("ac_30 = ", ac_30)
    print("ac_100 = ", 1000-ac_100)
