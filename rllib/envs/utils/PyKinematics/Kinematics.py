#!/usr/bin/env python

import os
import sys
from math import *

__current_path = os.path.dirname(__file__) or '.'
sys.path.insert(0, os.path.abspath(os.path.join(__current_path, os.path.pardir)))


class Kinematics(object):
    def __init__(self):
        self.__kineps = 1e-6

    def matrix2Euler(self, matrix, euler):
        sy = sqrt(matrix[0][0] * matrix[0][0] + matrix[1][0] * matrix[1][0])
        singular = sy < self.__kineps
        if singular:
            euler[1] = atan2(-matrix[2][0], 0)
            euler[0] = 0.0
            if matrix[2][0] > 0:
                euler[2] = atan2(-matrix[0][1], matrix[1][1])
            else:
                euler[2] = atan2(matrix[0][1], matrix[1][1])
        else:
            euler[1] = atan2(-matrix[2][0], sy)
            euler[2] = atan2(matrix[2][1], matrix[2][2])
            euler[0] = atan2(matrix[1][0], matrix[0][0])

    def euler2Matrix(self, euler, matrix):
        ca = cos(euler[0])
        sa = sin(euler[0])
        cb = cos(euler[1])
        sb = sin(euler[1])
        cc = cos(euler[2])
        sc = sin(euler[2])
        matrix[0][0] = ca * cb
        matrix[0][1] = ca * sb * sc - sa * cc
        matrix[0][2] = ca * sb * cc + sa * sc
        matrix[1][0] = sa * cb
        matrix[1][1] = sa * sb * sc + ca * cc
        matrix[1][2] = sa * sb * cc - ca * sc
        matrix[2][0] = -sb
        matrix[2][1] = cb * sc
        matrix[2][2] = cb * cc
        for i in range(3):
            for j in range(3):
                if fabs(matrix[i][j]) < self.__kineps:
                    matrix[i][j] = 0

    def harmonize(self, cmp_joints_rad, weight, tar_joints_rad, dof):
            dist_sqr = 0
            for i in range(dof):
                dist_sqr += pow(cmp_joints_rad[i] - tar_joints_rad[i], 2) * weight[i]
            return dist_sqr
