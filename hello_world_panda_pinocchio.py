import pathlib
import time
from pdb import set_trace

import numpy as np
import pybullet as p
import pybullet_data
from pinocchio.robot_wrapper import RobotWrapper


def main():
    client = p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)

    # Load plane
    planeID = p.loadURDF("plane.urdf")

    # Load Franka Panda Robot
    file_directory = str(pathlib.Path(__file__).parent.resolve())
    robot_URDF = file_directory + "/franka_panda/panda.urdf"
    robotID = p.loadURDF(robot_URDF, useFixedBase=True)

    # Build pin_robot
    robot = RobotWrapper.BuildFromURDF(robot_URDF)

    while True:
        p.stepSimulation()

    p.disconnect()


if __name__ == "__main__":
    main()
