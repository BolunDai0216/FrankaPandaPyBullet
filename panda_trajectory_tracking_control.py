import copy
import pathlib
import pickle
import time

import numpy as np
import pybullet as p
import pybullet_data
from pinocchio.robot_wrapper import RobotWrapper

from controllers.trajectory_tracking_control import TrajectoryTrackingController
from controllers.utils import get_state_update_pinocchio, send_joint_command


def main():
    p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    p.setTimeStep(1 / 240)
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)

    # Load plane
    p.loadURDF("plane.urdf")

    # Load Franka Panda Robot
    file_directory = str(pathlib.Path(__file__).parent.resolve())
    robot_URDF = file_directory + "/franka_panda/panda.urdf"
    robotID = p.loadURDF(robot_URDF, useFixedBase=True)

    # Build pin_robot
    robot = RobotWrapper.BuildFromURDF(robot_URDF)

    # Get frame ID for grasp target
    FRAME_ID = robot.model.getFrameId("panda_grasptarget")

    # Define desired end-effector position
    gt_desired_position = np.array([[-0.45], [-0.45], [1.0]])

    # Get active joint ids
    active_joint_ids = [0, 1, 2, 3, 4, 5, 6, 10, 11]

    # Disable the velocity control on the joints as we use torque control.
    p.setJointMotorControlArray(
        robotID, active_joint_ids, p.VELOCITY_CONTROL, forces=np.zeros(9),
    )

    planned_positions = []
    measured_positions = []
    time_array = []

    for i in range(10000):
        # Update pinocchio model and get joint states
        q, dq = get_state_update_pinocchio(robot, robotID)

        # Get end-effector position
        _gt_position = robot.data.oMf[FRAME_ID].translation
        gt_position = _gt_position[:, np.newaxis]

        # Get initial position and create trajectory
        if i == 0:
            gt_init_position = copy.deepcopy(gt_position)
            controller = TrajectoryTrackingController(
                gt_init_position, gt_desired_position, 20.0
            )
            controller.save_plan()

        # Get joint torques using a resolved rate controller
        t = i * (1 / 240)
        tau, planned_pos = controller.get_control(robot, t, q, dq, FRAME_ID)

        # Send joint commands to motor
        send_joint_command(robotID, tau)

        if i % 500 == 0:
            print(f"End-effector position: {gt_position}")

        p.stepSimulation()
        time.sleep(1e-4)

        planned_positions.append(planned_pos)
        measured_positions.append(copy.deepcopy(gt_position))
        time_array.append(t)

    p.disconnect()

    position_data = {
        "planned": planned_positions,
        "measured": measured_positions,
        "time": time_array,
    }

    with open("data/data.pickle", "wb") as handle:
        pickle.dump(position_data, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    main()
