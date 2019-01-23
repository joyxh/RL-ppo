#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @File :SixKin.py
from envs.utils.PyKinematics.Kinematics import *
import numpy as np

__current_path = os.path.dirname(__file__) or '.'
sys.path.insert(0, os.path.abspath(os.path.join(__current_path, os.path.pardir)))


class SixAxisKin(Kinematics):
    def __init__(self, a, d, rot_dir):
        super(SixAxisKin, self).__init__()
        self.__a = a
        self.__d = d
        self.__rot_dir = rot_dir
        self.__min_limit = np.zeros(6)
        self.__max_limit = np.zeros(6)
        self.__solution_num = 0
        self.__kineps = 1e-6
        self.__alpha = [0.0, -0.5 * pi, 0.0, -0.5 * pi, 0.5 * pi, -0.5 * pi]
        self.__MAX = 100000000000.0
        self.__end_effector = np.zeros(3)
        self.__euler = np.zeros(3)
        self.__x = 0.0
        self.__y = 0.0
        self.__z = 0.0

    def setLimit(self, min_limit_rad, max_limit_rad):
        for i in range(6):
            self.__min_limit[i] = min_limit_rad[i]
            self.__max_limit[i] = max_limit_rad[i]

    def forwardKin(self, joint_rad, coord_rad):
        c = np.zeros(6)
        s = np.zeros(6)
        for i in range(6):
            if i != 1:
                s[i] = sin(joint_rad[i]) * self.__rot_dir[i]
                c[i] = cos(joint_rad[i])
            else:
                s[i] = -cos(joint_rad[i])
                c[i] = sin(joint_rad[i]) * self.__rot_dir[i]
        trans_matrix_total = np.eye(4)
        trans_matrix = np.zeros((6, 4, 4))

        for i in range(6):
            trans_matrix[i][0][0] = c[i]                                   # 1 1
            trans_matrix[i][0][1] = -s[i]                                  # 1 2
            trans_matrix[i][0][2] = 0.0                                    # 1 3
            trans_matrix[i][0][3] = self.__a[i]                            # 1 4
            trans_matrix[i][1][0] = s[i] * cos(self.__alpha[i])            # 2 1
            trans_matrix[i][1][1] = c[i] * cos(self.__alpha[i])            # 2 2
            trans_matrix[i][1][2] = -sin(self.__alpha[i])                  # 2 3
            trans_matrix[i][1][3] = -sin(self.__alpha[i])*self.__d[i]      # 2 4
            trans_matrix[i][2][0] = s[i]*sin(self.__alpha[i])              # 3 1
            trans_matrix[i][2][1] = c[i]*sin(self.__alpha[i])              # 3 2
            trans_matrix[i][2][2] = cos(self.__alpha[i])                   # 3 3
            trans_matrix[i][2][3] = cos(self.__alpha[i])*self.__d[i]       # 3 4
            trans_matrix[i][3][0] = 0.0                                    # 4 1
            trans_matrix[i][3][1] = 0.0                                    # 4 2
            trans_matrix[i][3][2] = 0.0                                    # 4 3
            trans_matrix[i][3][3] = 1.0                                    # 4 4
        for m in range(6):
            trans_matrix_total = np.dot(trans_matrix_total, trans_matrix[m])

        super(SixAxisKin, self).matrix2Euler(trans_matrix_total[:3, :3], coord_rad[3:])
        coord_rad[0] = trans_matrix_total[0][3]
        coord_rad[1] = trans_matrix_total[1][3]
        coord_rad[2] = trans_matrix_total[2][3]

    def inverseKin(self, coord_rad, all_solution_rad):
        self.calCoordOriginXYZ(coord_rad)
        self.calJoint1and3(all_solution_rad)
        self.calJoint2(all_solution_rad)
        self.calLeaveJoints(all_solution_rad)
        self.jointsTransform(all_solution_rad)
        return self.__solution_num

    def calCoordOriginXYZ(self, coord_rad):
        for i in range(3):
            self.__euler[i] = coord_rad[i + 3]
            self.__end_effector[i] = coord_rad[i]
        sin_euler = list(map(sin, self.__euler))
        cos_euler = list(map(cos, self.__euler))
        self.__x = self.__end_effector[0] - (cos_euler[0] * sin_euler[1] * cos_euler[2] + sin_euler[0] * sin_euler[2]) * self.__d[5]
        self.__y = self.__end_effector[1] - (sin_euler[0] * sin_euler[1] * cos_euler[2] - cos_euler[0] * sin_euler[2]) * self.__d[5]
        self.__z = self.__end_effector[2] - (cos_euler[1] * cos_euler[2]) * self.__d[5] - self.__d[0]

    def calJoint1and3(self, all_solution_rad):
        delta_joint1 = pow(self.__x, 2) + pow(self.__y, 2) - pow(self.__d[2], 2)
        pos_neg = [1, -1]
        if delta_joint1 < 0:
            print("calJoint1, NO solution")
            return
        if sqrt(delta_joint1) < self.__kineps:
            joint1_rad = atan2(self.__y, self.__x) - 0.5 * pi
            K = (pow(self.__x, 2) + pow(self.__y, 2) + pow(self.__z, 2) + pow(self.__a[1], 2) - pow(self.__a[2], 2) -
                 pow(self.__a[3], 2) - 2.0 * self.__a[1] * (cos(joint1_rad) * self.__x + sin(joint1_rad) * self.__y) -
                 pow(self.__d[2], 2) - pow(self.__d[3], 2)) * 0.5 / self.__a[2]
            delta_joint3 = pow(self.__a[3], 2) + pow(self.__d[3], 2) - pow(K, 2)
            if delta_joint3 < 0:
                print("delta_joint1 = 0, delta_joint3 < 0, NO solution")
                return
            elif fabs(delta_joint3) < self.__kineps:
                all_solution_rad[0 * 6 + 2] = atan2(self.__a[3], self.__d[3]) - atan2(K, 0.0)
                self.__solution_num += 1
                return
            else:
                self.__solution_num += 2
                all_solution_rad[0 * 6] = joint1_rad
                all_solution_rad[1 * 6] = joint1_rad
                all_solution_rad[0 * 6 + 1] = atan2(self.__a[3], self.__d[3]) - atan2(K, sqrt(delta_joint3))
                all_solution_rad[1 * 6 + 1] = atan2(self.__a[3], self.__d[3]) - atan2(K, -sqrt(delta_joint3))
        else:
            has_solution = False
            for i in range(2):
                joint1_rad = atan2(self.__y, self.__x) - atan2(self.__d[2], pos_neg[i] * sqrt(delta_joint1))
                K = (pow(self.__x, 2) + pow(self.__y, 2) + pow(self.__z, 2) + pow(self.__a[1], 2) - pow(self.__a[2], 2)
                     - pow(self.__a[3], 2) - 2.0 * self.__a[1] * (cos(joint1_rad) * self.__x + sin(joint1_rad) *
                                                                  self.__y) - pow(self.__d[2], 2) - pow(self.__d[3], 2)) * 0.5 / self.__a[2]
                delta_joint3 = pow(self.__a[3], 2) + pow(self.__d[3], 2) - pow(K, 2)
                if delta_joint3 < 0:
                    break
                if fabs(delta_joint3) < self.__kineps:
                    all_solution_rad[self.__solution_num * 6] = joint1_rad
                    all_solution_rad[self.__solution_num * 6 + 2] = atan2(self.__a[3], self.__d[3]) - atan2(K, 0.0)
                    self.__solution_num += 1
                    has_solution = True
                else:
                    all_solution_rad[self.__solution_num * 6] = joint1_rad
                    all_solution_rad[self.__solution_num * 6 + 2] = atan2(self.__a[3], self.__d[3]) - atan2(K, sqrt(delta_joint3))
                    self.__solution_num += 1
                    all_solution_rad[self.__solution_num * 6] = joint1_rad
                    all_solution_rad[self.__solution_num * 6 + 2] = atan2(self.__a[3], self.__d[3]) - atan2(K, -sqrt(delta_joint3))
                    self.__solution_num += 1
                    has_solution = True
            if not has_solution:
                print("calJoint1and3, NO solution")

    def calJoint2(self, all_solution_rad):
        for i in range(self.__solution_num):
            temp1 = cos(all_solution_rad[6 * i]) * self.__x + sin(all_solution_rad[6 * i]) * self.__y - self.__a[1]
            temp2 = self.__a[3] + self.__a[2] * cos(all_solution_rad[i * 6 + 2])
            temp3 = -self.__d[3] + self.__a[2] * sin(all_solution_rad[i * 6 + 2])
            s23 = -temp2 * self.__z + temp1 * temp3
            c23 = temp3 * self.__z + temp1 * temp2
            joint_23 = atan2(s23, c23)
            all_solution_rad[i * 6 + 1] = joint_23 - all_solution_rad[i * 6 + 2]

    def calLeaveJoints(self, all_solution_rad):
        count = self.__solution_num
        R30 = np.zeros((3, 3))
        R06 = np.zeros((3, 3))
        for i in range(count):
            first_three_joints = np.array([all_solution_rad[i * 6], all_solution_rad[i * 6 + 1],
                                           all_solution_rad[i * 6 + 2]], dtype=float)
            c = list(map(cos, first_three_joints))
            s = list(map(sin, first_three_joints))
            R30[0][0] = c[0] * c[1] * c[2] - c[0] * s[1] * s[2]
            R30[0][1] = s[0] * c[1] * c[2] - s[0] * s[1] * s[2]
            R30[0][2] = -c[1] * s[2] - s[1] * c[2]
            R30[1][0] = -c[0] * c[1] * s[2] - c[0] * s[1] * c[2]
            R30[1][1] = -s[0] * c[1] * s[2] - s[0] * s[1] * c[2]
            R30[1][2] = -c[1] * c[2] + s[1] * s[2]
            R30[2][0] = -s[0]
            R30[2][1] = c[0]
            R30[2][2] = 0.0

            super(SixAxisKin, self).euler2Matrix(self.__euler, R06)
            R36 = np.dot(R30, R06)
            if fabs(R36[1][2]) > 1 - self.__kineps:
                all_solution_rad[i * 6 + 4] = 0
                all_solution_rad[i * 6 + 3] = 0
                all_solution_rad[i * 6 + 5] = atan2(-R36[2][0], R36[0][0])
            else:
                all_solution_rad[i * 6 + 4] = acos(R36[1][2])
                s5 = sin(all_solution_rad[i * 6 + 4])
                all_solution_rad[i * 6 + 3] = atan2(R36[2][2] / s5, -R36[0][2] / s5)
                all_solution_rad[i * 6 + 5] = atan2(-R36[1][1] / s5, R36[1][0] / s5)

                all_solution_rad[self.__solution_num * 6] = all_solution_rad[i * 6]
                all_solution_rad[self.__solution_num * 6 + 1] = all_solution_rad[i * 6 + 1]
                all_solution_rad[self.__solution_num * 6 + 2] = all_solution_rad[i * 6 + 2]
                all_solution_rad[self.__solution_num * 6 + 4] = - all_solution_rad[i * 6 + 4]
                s5 = sin(all_solution_rad[self.__solution_num * 6 + 4])
                all_solution_rad[self.__solution_num * 6 + 3] = atan2(R36[2][2] / s5, -R36[0][2] / s5)
                all_solution_rad[self.__solution_num * 6 + 5] = atan2(-R36[1][1] / s5, R36[1][0] / s5)
                self.__solution_num += 1

    def jointsTransform(self, all_solution_rad):
        for i in range(self.__solution_num):
            all_solution_rad[i * 6 + 1] += 0.5 * pi
            for j in range(6):
                all_solution_rad[i * 6 + j] *= self.__rot_dir[j]
                if fabs(all_solution_rad[i * 6 + j]) < self.__kineps:
                    all_solution_rad[i * 6 + j] = 0
                if all_solution_rad[i * 6 + j] > pi:
                    all_solution_rad[i * 6 + j] -= 2.0 * pi
                elif all_solution_rad[i * 6 + j] < -pi:
                    all_solution_rad[i * 6 + j] += 2.0 * pi

    def checkSolution(self, all_solution_rad, index, ref_joints_rad):
        min_dist = self.__MAX
        min_index = 10
        temp_joint_rad = np.zeros(3)
        for i in range(6):
            if self.__max_limit[i] > pi:
                temp_joint_rad[0] = all_solution_rad[index * 6 + i] + 2.0 * pi
                if self.__min_limit[i] < temp_joint_rad[0] < self.__max_limit[i]:
                    dist = fabs(temp_joint_rad[0] - ref_joints_rad[i])
                    if dist < min_dist:
                        min_dist = dist
                        min_index = 0
            elif self.__min_limit[i] < -pi:
                temp_joint_rad[1] = all_solution_rad[index * 6 + i] - 2.0 * pi
                if self.__min_limit[i] < temp_joint_rad[1] < self.__max_limit[i]:
                    dist = fabs(temp_joint_rad[1] - ref_joints_rad[i])
                    if dist < min_dist:
                        min_dist = dist
                        min_index = 1
            temp_joint_rad[2] = all_solution_rad[index * 6 + i]
            if self.__min_limit[i] < temp_joint_rad[2] < self.__max_limit[i]:
                dist = fabs(temp_joint_rad[2] - ref_joints_rad[i])
                if dist < min_dist:
                    min_index = 2
            if min_index > 2:
                return -1
            else:
                all_solution_rad[index * 6 + i] = temp_joint_rad[min_index]
        return 0

    def ikSolution(self, coord_rad, weight, ref_joints_rad, result_joints_rad, hand_mode):
        AboveHand = 0
        BelowHand = 1
        all_solution_rad = np.zeros(48)
        min_dist = self.__MAX
        min_index = 10
        dof = 6
        self.__solution_num = 0
        solution_num = self.inverseKin(coord_rad, all_solution_rad)
        # print("solution: ", solution_num)
        # print(np.rad2deg(all_solution_rad).reshape(-1, 6))
        if 0 == solution_num:
            return -1
        for i in range(solution_num):
            one_solution_rad = np.zeros(6)
            for j in range(dof):
                one_solution_rad[j] = all_solution_rad[i * 6 + j]
            if hand_mode == AboveHand:
                if all_solution_rad[i * 6 + 2] < -0.5 * pi:
                    continue
                if self.checkSolution(all_solution_rad, i, ref_joints_rad) < 0:
                    continue
                dist = super(SixAxisKin, self).harmonize(one_solution_rad, weight, ref_joints_rad, dof)
                if dist < min_dist:
                    min_dist = dist
                    min_index = i
            elif hand_mode == BelowHand:
                if all_solution_rad[i * 6 + 2] > -0.5 * pi:
                    continue
                if self.checkSolution(all_solution_rad, i, ref_joints_rad) < 0:
                    continue
                dist = super(SixAxisKin, self).harmonize(one_solution_rad, weight, ref_joints_rad, dof)
                if dist < min_dist:
                    min_dist = dist
                    min_index = i
            else:
                if self.checkSolution(all_solution_rad, i, ref_joints_rad) < 0:
                    continue
                dist = super(SixAxisKin, self).harmonize(one_solution_rad, weight, ref_joints_rad, dof)
                if dist < min_dist:
                    min_dist = dist
                    min_index = i
        if 10 == min_index:
            return -2
        else:
            for k in range(dof):
                result_joints_rad[k] = all_solution_rad[min_index * 6 + k]
            return 0


if __name__ == '__main__':
    np.set_printoptions(suppress=True)
    max_jnt = np.array([150.0, 70.0, 60.0, 210.0, 115.0, 360.0], np.float)
    min_jnt = np.array([-150.0, -145.0, -150.0, -210.0, -115.0, -360.0], np.float)
    rot_dir = np.array([-1.0, -1.0, 1.0, -1.0, 1.0, -1.0], np.float)
    a = np.array([0.0, 76.0, 210.0, 65.0, 0.0, 0.0], np.float)
    d = np.array([303.3, 0.0, 0.0, 245.0, -0.0, 160.0], np.float)
    weight = np.array([5.0, 3.0, 2.0, 1.0, 0.5, 0.1], np.float)
    ref_joints_deg = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], np.float)
    ref_joints_rad = np.zeros(6)
    result_joints_rad = np.zeros(6)

    kin = SixAxisKin(a, d, rot_dir)
    kin.setLimit(min_jnt, max_jnt)
    input_jnt = np.array([40.0, 50.0, 60.0, 70.0, 80.0, 90.0], np.float)
    input_jnt_rad = np.deg2rad(input_jnt)
    out_coord = np.zeros(6)
    kin.forwardKin(input_jnt_rad, out_coord)
    print(out_coord)

    # all_jnt = np.zeros(48)
    # kin.inverseKin(out_coord, all_jnt)
    # print "all_jnt: \n", np.rad2deg(all_jnt)

    ref_value = kin.ikSolution(out_coord, weight, ref_joints_rad, result_joints_rad, 2)
    print(ref_value)
    print(np.rad2deg(result_joints_rad))

