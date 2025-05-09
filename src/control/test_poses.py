import os
import glob
import yaml

import numpy as np

from typing import Dict, Any

from pybullet_object_models import ycb_objects  # type:ignore

from src.simulation import Simulation

import pybullet as p
from src.control import IKSolver
from scipy.spatial.transform import Rotation as R


def run_exp(config: Dict[str, Any]):
    # Example Experiment Runner File
    print("Simulation Start:")
    print(config['world_settings'], config['robot_settings'])
    object_root_path = ycb_objects.getDataPath()
    files = glob.glob(os.path.join(object_root_path, "Ycb*"))
    obj_names = [file.split('/')[-1] for file in files]
    sim = Simulation(config)
    for obj_name in obj_names:
        for tstep in range(10):
            sim.reset(obj_name)
            print((f"Object: {obj_name}, Timestep: {tstep},"
                   f" pose: {sim.get_ground_tuth_position_object}"))
            pos, ori = sim.robot.pos, sim.robot.ori
            print(f"Robot inital pos: {pos} orientation: {ori}")
            l_lim, u_lim = sim.robot.lower_limits, sim.robot.upper_limits
            print(f"Robot Joint Range {l_lim} -> {u_lim}")
            sim.robot.print_joint_infos()
            jpos = sim.robot.get_joint_positions()
            print(f"Robot current Joint Positions: {jpos}")
            jvel = sim.robot.get_joint_velocites()
            print(f"Robot current Joint Velocites: {jvel}")
            ee_pos, ee_ori = sim.robot.get_ee_pose()
            print(f"Robot End Effector Position: {ee_pos}")
            print(f"Robot End Effector Orientation: {ee_ori}")
            inverse_kinematics = IKSolver(sim)

            target_pos, target_ori = p.getBasePositionAndOrientation(sim.object.id)
            print(f"Target Pos: {target_pos}")
            print(f"Target Ori: {target_ori}")
            target_pos = np.array([-0.05018395, -0.46971428, 1.4])
            target_pos = np.array([0, -0.65, 1.8])
            target_ori = R.from_euler('xyz', [np.pi, 0, 0]).as_quat()
            q = inverse_kinematics.compute_target_configuration(target_pos, target_ori)
            #q = p.calculateInverseKinematics(sim.robot.id, sim.robot.ee_idx, target_pos, target_ori)[:-2]
            print(f"New Joint Configuration: {q}")
            for i in range(10000):
                if q is not None:
                    sim.robot.position_control(q)
                sim.step()
                if i > 1:
                    print("Base POSE ! ", p.getJointStates(sim.robot.id, sim.robot.arm_idx))
                    print("CHECK joint limits ! ", sim.robot.get_joint_limits())
                    print("EE POSE ! ", sim.robot.get_ee_pose())
                    print(f"Target Orientation: {target_ori}")
                    print(R.from_quat(sim.robot.get_ee_pose()[1]).as_euler('xyz', degrees=True))
                # for getting renders
                # rgb, depth, seg = sim.get_ee_renders()
                # rgb, depth, seg = sim.get_static_renders()
                obs_position_guess = np.zeros((2, 3))
                print((f"[{i}] Obstacle Position-Diff: "
                       f"{sim.check_obstacle_position(obs_position_guess)}"))
                goal_guess = np.zeros((7,))
                print((f"[{i}] Goal Obj Pos-Diff: "
                       f"{sim.check_goal_obj_pos(goal_guess)}"))
                print(f"[{i}] Goal Satisfied: {sim.check_goal()}")
    sim.close()


if __name__ == "__main__":
    with open("../../configs/test_config.yaml", "r") as stream:
        try:
            config = yaml.safe_load(stream)
            print(config)
        except yaml.YAMLError as exc:
            print(exc)
    run_exp(config)
