import time

import pybullet as p
import pybullet_data
import numpy as np


def main():
    client = p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, 0)

    planeID = p.loadURDF("plane.urdf")
    robotID = p.loadURDF("./franka_panda/panda.urdf", useFixedBase=True)

    n_j = p.getNumJoints(robotID)

    debug_sliders = []
    joint_ids = []
    for i in range(n_j):
        _joint_infos = p.getJointInfo(robotID, i)
        if _joint_infos[2] != p.JOINT_FIXED:
            debug_sliders.append(
                p.addUserDebugParameter(
                    _joint_infos[1].decode("UTF-8"),
                    _joint_infos[8],
                    _joint_infos[9],
                    0.0,
                )
            )
            joint_ids.append(_joint_infos[0])

    while True:
        for slider_id, joint_id in zip(debug_sliders, joint_ids):
            _joint_angle = p.readUserDebugParameter(slider_id)
            p.resetJointState(robotID, joint_id, _joint_angle)

        p.stepSimulation()

    p.disconnect()


if __name__ == "__main__":
    main()
