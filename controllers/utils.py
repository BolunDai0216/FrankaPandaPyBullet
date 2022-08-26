from pdb import set_trace

import numpy as np
import pybullet as p

# For Franka Panda these are the active joints
ACTIVE_JOINT_IDS = [0, 1, 2, 3, 4, 5, 6, 10, 11]


def get_state(robot, robotID):
    q = np.zeros(9)
    dq = np.zeros(9)

    for i, id in enumerate(ACTIVE_JOINT_IDS):
        _joint_state = p.getJointState(robotID, id)
        q[i], dq[i] = _joint_state[0], _joint_state[1]

    return q, dq


def update_pinocchio(robot, q, dq):
    robot.computeJointJacobians(q)
    robot.framesForwardKinematics(q)
    robot.centroidalMomentum(q, dq)


def get_state_update_pinocchio(robot, robotID):
    q, dq = get_state(robot, robotID)
    update_pinocchio(robot, q, dq)

    return q, dq


def send_joint_command(robotID, tau):
    zeroGains = tau.shape[0] * (0.0,)

    p.setJointMotorControlArray(
        robotID,
        ACTIVE_JOINT_IDS,
        p.TORQUE_CONTROL,
        forces=tau,
        positionGains=zeroGains,
        velocityGains=zeroGains,
    )


def compute_quat_vec_error(quat_1, quat_2):
    eta1 = quat_1[-1]
    eta2 = quat_2[-1]

    quat_vec1 = quat_1[:3]
    quat_vec2 = quat_2[:3]

    delta_quat_vec = (
        eta1 * quat_vec2 - eta2 * quat_vec1 - np.cross(quat_vec1, quat_vec2)
    )
    return delta_quat_vec
