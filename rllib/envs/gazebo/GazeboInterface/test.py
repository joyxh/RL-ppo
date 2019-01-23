from RobotControl import RobotControl
import pickle
import time

robot = RobotControl("rr_arm", 6)

robot.setSpeedRate(100)

# robot.setJointPos([0,-31,-157,0,0,0], wait=True)
# print(robot.getRobotPos())
# time.sleep(2)
#
# robot.setJointPos([0,-30,-186,0,0,0], wait=True)
# print(robot.getRobotPos())
# time.sleep(2)
#
# robot.setJointPos([0,108,-24.7,0,0,0], wait=True)
# print(robot.getRobotPos())
# time.sleep(2)

# robot.setJointPos([0,0,-206,0,0,0], wait=True)
# print(robot.getRobotPos())

def get_pose():
    step = 2
    limit = (-50.9, 126.04)
    len = int((limit[1]-limit[0]) / step)
    robot_pos = []
    for i in range(len):
        degree = limit[0] + step * i
        robot.setJointPos([0, degree, -90, 0, 0, 0], wait=True)
        time.sleep(0.1)
        robot_pos.append(robot.getRobotPos())

    with open('circle.txt', 'wb') as file:
        pickle.dump(robot_pos, file)

if __name__ == '__main__':
    get_pose()