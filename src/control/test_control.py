import os
import glob
import yaml

import numpy as np
from scipy.spatial.transform import Rotation as R

from typing import Dict, Any

from pybullet_object_models import ycb_objects  # type:ignore

from src.simulation import Simulation

import pybullet as p
from control import IKSolver

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
            
            # target_pos, target_ori = p.getBasePositionAndOrientation(sim.object.id)
            
            # target_pos = np.array([-0.05018395, -0.46971428,  1.4 ])
            # q = inverse_kinematics.compute_target_configuration(target_pos, target_ori)
            object_views = {
                'top': {'pos': [0, -0.65, 1.75], 'ori':[np.pi, 0, 0]},
                'right': {'pos': [0, -0.3, 1.75], 'ori': [3/4 * np.pi, 0, 0]},
                'left': {'pos': [0, -0.9, 1.75], 'ori': [5/4 * np.pi, 0, 0]}, # initially 5/4 * np.pi
                'front': {'pos': [0.3, -0.65, 1.75], 'ori': [3/4 * np.pi, 0, -np.pi / 2]}, # [0, -3/4 * np.pi, 0]
                'back': {'pos': [-0.3, -0.65, 1.75], 'ori': [3/4 * np.pi, 0, np.pi / 2]} # (0, 3/4 * np.pi, 0]
            }
            
            for key,item in object_views.items():
                print(f"Solve for {key}")
                pos = item['pos']
                ori = R.from_euler('xyz', item['ori']).as_quat()
                q = inverse_kinematics.compute_target_configuration(pos, ori)



            print(f"New Joint Configuration: {q}")
            for i in range(10000):
                if q is not None:
                    sim.robot.position_control(q)
                sim.step()
                if i > 1:
                    print("Base POSE ! ", p.getJointStates(sim.robot.id, sim.robot.arm_idx))
                    print("CHECK joint limits ! ", sim.robot.get_joint_limits())
                    print("EE POSE ! ", sim.robot.get_ee_pose())
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
