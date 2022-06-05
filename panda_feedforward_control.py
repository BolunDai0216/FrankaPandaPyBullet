import copy
import pathlib
from pdb import set_trace

import numpy as np
import pybullet as p
import pybullet_data
from pinocchio.robot_wrapper import RobotWrapper

from controllers.impedance_control import impedance_control
from controllers.utils import get_state_update_pinocchio, send_joint_command
from controllers.feedforward_control import FeedforwardController


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

    # Get frame ID for grasp target
    GRASPTARGET_ID = robot.model.getFrameId("panda_grasptarget")

    # Define desired end-effector position
    gt_desired_position = np.array([0.3, 0.3, 0.9])

    # Get active joint ids
    active_joint_ids = [0, 1, 2, 3, 4, 5, 6, 10, 11]

    # Disable the velocity control on the joints as we use torque control.
    p.setJointMotorControlArray(
        robotID,
        active_joint_ids,
        p.VELOCITY_CONTROL,
        forces=np.zeros(9),
    )

    for i in range(15000):
        # Update pinocchio model and get joint states
        joint_angles, joint_velocities = get_state_update_pinocchio(robot, robotID)

        # Get end-effector position
        gt_position = robot.data.oMf[GRASPTARGET_ID].translation

        if i == 0:
            gt_init_position = copy.deepcopy(gt_position)
            ff_control = FeedforwardController(
                gt_init_position[:, np.newaxis], gt_desired_position[:, np.newaxis], 5.0
            )
            ff_control.save_plan()

        # Get joint torques using a feedforward controller
        tau = impedance_control(
            robot, GRASPTARGET_ID, gt_desired_position, joint_angles, joint_velocities
        )

        # Send joint commands to motor
        send_joint_command(robotID, tau)

        if i % 500 == 0:
            print(f"End-effector position: {gt_position}")

        p.stepSimulation()

    p.disconnect()


if __name__ == "__main__":
    main()
