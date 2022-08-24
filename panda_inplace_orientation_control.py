import copy
import pathlib
import pickle
import time
from pdb import set_trace

import numpy as np
import pinocchio as pin
import pybullet as p
import pybullet_data
from pinocchio.robot_wrapper import RobotWrapper
from scipy.spatial.transform import Rotation as R

from controllers.utils import get_state_update_pinocchio, send_joint_command


def compute_quat_vec_error(quat_1, quat_2):
    eta1 = quat_1[0]
    eta2 = quat_2[0]

    quat_vec1 = quat_1[1:]
    quat_vec2 = quat_2[1:]

    delta_quat_vec = (
        eta1 * quat_vec2 - eta2 * quat_vec1 - np.cross(quat_vec1, quat_vec2)
    )

    return delta_quat_vec


def main():
    p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, 0)
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

    # Create target location indicator
    visualShapeId = p.createVisualShape(
        shapeType=p.GEOM_SPHERE, radius=0.04, rgbaColor=[1, 0, 0, 1]
    )
    p.createMultiBody(
        baseVisualShapeIndex=visualShapeId,
        basePosition=gt_desired_position[:, 0].tolist(),
    )

    i = 0

    while True:
        # Update pinocchio model and get joint states
        q, dq = get_state_update_pinocchio(robot, robotID)

        # Get end-effector position
        _gt_rotation = robot.data.oMf[FRAME_ID].rotation
        gt_rotation = R.from_matrix(_gt_rotation)
        gt_quat = gt_rotation.as_quat()

        if i == 0:
            target_rotation = R.from_euler("z", 1, degrees=True)
            target_orientation = R.from_matrix(
                target_rotation.as_matrix() @ _gt_rotation
            )
            target_quat = copy.deepcopy(target_orientation.as_quat())
            target_position = copy.deepcopy(robot.data.oMf[FRAME_ID].translation)

        # Get frame ID for grasp target
        jacobian_frame = pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
        # jacobian_frame = pin.ReferenceFrame.LOCAL

        # Get Jacobian from grasp target frame
        # preprocessing is done in get_state_update_pinocchio()
        jacobian = robot.getFrameJacobian(FRAME_ID, jacobian_frame)

        # Get pseudo-inverse of frame Jacobian
        pinv_jac_t = np.linalg.pinv(jacobian[:3, :])
        pinv_jac_r = np.linalg.pinv(jacobian[3:, :])
        pinv_jac = np.linalg.pinv(jacobian)

        # Compute Coriolis and Gravitational terms
        C = robot.nle(q, dq)

        # Get frame position and velocity
        position = robot.data.oMf[FRAME_ID].translation
        velocity = jacobian[:3, :] @ dq

        # Get frame error
        delta_p = target_position - position
        delta_v = -velocity

        quat_vec_error = compute_quat_vec_error(target_quat, gt_quat)

        target_dx = np.array([[0.0], [0.0], [0.0], [0.00], [0.00], [-0.01]])

        # Get joint torques using a resolved rate controller
        # tau = (
        #     2 * (pinv_jac @ target_dx - dq[:, np.newaxis])
        #     + C[:, np.newaxis]
        #     - 0.1 * dq[:, np.newaxis]
        # )
        tau = 0.5 * (pinv_jac @ target_dx - dq[:, np.newaxis])

        # Send joint commands to motor
        send_joint_command(robotID, tau)

        # if i % 500 == 0:
        #     print(f"End-effector position: {quat_vec_error}")

        p.stepSimulation()
        time.sleep(1e-4)
        i += 1

    p.disconnect()


if __name__ == "__main__":
    main()
