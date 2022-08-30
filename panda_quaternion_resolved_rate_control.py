import copy
import pathlib
import time
from pdb import set_trace

import numpy as np
import numpy.linalg as LA
import pinocchio as pin
import pybullet as p
import pybullet_data
from pinocchio.robot_wrapper import RobotWrapper
from scipy.spatial.transform import Rotation as R

from controllers.utils import (
    compute_quat_vec_error,
    get_state_update_pinocchio,
    send_joint_command,
)


def main():
    """
    This file implements an operational space controller proportional controller.

    The part of the controller that controls the position is:
    τ_p = J_p^+[K_{p, p}(x_des - x)]

    The part of the controller that controls the orientation is:
    τ_o = J_o^+[K_{p, φ}(φ_des - φ)]

    where φ is the vector part of a quaternion.

    Then, the final controller is:
    τ = τ_p + τ_o + G(q)
    """
    p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
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

    # Get active joint ids
    active_joint_ids = [0, 1, 2, 3, 4, 5, 6, 10, 11]

    # Disable the velocity control on the joints as we use torque control.
    p.setJointMotorControlArray(
        robotID, active_joint_ids, p.VELOCITY_CONTROL, forces=np.zeros(9),
    )

    target_joint_angles = [
        0.0,
        -0.785398163,
        0.0,
        -2.35619449,
        0.0,
        1.57079632679,
        0.785398163397,
        0.001,
        0.001,
    ]

    for i, joint_ang in enumerate(target_joint_angles):
        p.resetJointState(robotID, active_joint_ids[i], joint_ang, 0.0)

    init = True
    target_position = np.array([[0.3], [-0.4], [0.5]])

    for i in range(80000):
        # Update pinocchio model and get joint states
        q, dq = get_state_update_pinocchio(robot, robotID)

        # Get end-effector position
        _gt_position = robot.data.oMf[FRAME_ID].translation
        gt_position = _gt_position[:, np.newaxis]

        # Get target orientation based on initial orientation
        if init:
            _init_rotation = robot.data.oMf[FRAME_ID].rotation  # R10
            _target_rotation = (
                R.from_euler("x", 0, degrees=True).as_matrix()
                @ R.from_euler("z", 90, degrees=True).as_matrix()
                @ _init_rotation
            )  # R20
            target_rotation = R.from_matrix(_target_rotation)
            target_quaternion = copy.deepcopy(target_rotation.as_quat())
            init = False

        # Get end-effector orientation
        _gt_orientation = robot.data.oMf[FRAME_ID].rotation
        _gt_quaternion = R.from_matrix(_gt_orientation).as_quat()

        # Orientation error in quaternion form
        quat_err = compute_quat_vec_error(target_quaternion, _gt_quaternion)

        set_trace()
        R_err = (
            target_rotation.as_matrix() @ R.from_matrix(_gt_orientation).as_matrix().T
        )
        rotvec_err = R.from_matrix(R_err).as_rotvec()

        # Get frame ID for grasp target
        jacobian_frame = pin.ReferenceFrame.LOCAL_WORLD_ALIGNED

        # Get Jacobian from grasp target frame
        # preprocessing is done in get_state_update_pinocchio()
        jacobian = robot.getFrameJacobian(FRAME_ID, jacobian_frame)

        # Get pseudo-inverse of frame Jacobian
        pinv_jac = np.linalg.pinv(jacobian)

        # Compute Gravitational terms
        G = robot.gravity(q)

        # Compute controller
        target_dx = np.zeros((6, 1))
        target_dx[:3] = 1.0 * (target_position - gt_position)
        target_dx[3:] = np.diag([3.0, 3.0, 3.0]) @ quat_err[:, np.newaxis]

        tau = (pinv_jac @ target_dx - dq[:, np.newaxis]) + G[:, np.newaxis]

        # Set control for the two fingers to zero
        tau[-1] = 0.0
        tau[-2] = 0.0

        # Send joint commands to motor
        send_joint_command(robotID, tau)
        # time.sleep(1e-2)

        if i % 500 == 0:
            print(
                "Iter {:.2e} \t ǁeₒǁ₂: {:.2e} \t ǁeₚǁ₂: {:.2e}".format(
                    i, LA.norm(quat_err), LA.norm(target_position - gt_position)
                ),
            )

        p.stepSimulation()

    p.disconnect()


if __name__ == "__main__":
    main()
