import os
import glob
import yaml

import numpy as np
from scipy.spatial.transform import Rotation as R

from typing import Dict, Any

from pybullet_object_models import ycb_objects  # type:ignore

from src.simulation import Simulation

import pybullet as p
from src.control import IKSolver

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
                #'test': {'pos': [0.0, -0.3, 1.18], 'ori':[np.pi, 0, 0]},
                'top': {'pos': [0, -0.60, 1.5], 'ori':[np.pi, 0, 0]},
                'right': {'pos': [0, -0.3, 1.5], 'ori': [7/8 * np.pi, 0, 0]},
                'left': {'pos': [0, -0.6, 1.5], 'ori': [9/8 * np.pi, 0, 0]}, # initially 5/4 * np.pi
                'front': {'pos': [0.3, -0.60, 1.5], 'ori': [3/4 * np.pi, 0, -np.pi / 2]}, # [0, -3/4 * np.pi, 0]
                'back': {'pos': [-0.3, -0.60, 1.5], 'ori': [3/4 * np.pi, 0, np.pi / 2]} # (0, 3/4 * np.pi, 0]

                # 'top': {'pos': (0, -0.6, 1.8), 'ori': (np.pi, 0, 0)},
                # 'right': {'pos': (0, -0.3, 1.72), 'ori': (7/8 * np.pi, 0, 0)}, # initially 3/4 * np.pi
                # 'front': {'pos': (0.4, -0.6, 1.72), 'ori': (7 / 8 * np.pi, 0, -np.pi / 2)},  # (0, -3/4 * np.pi, 0)
                # 'left': {'pos': (0, -0.9, 1.72), 'ori': (9/8 * np.pi, 0, 0)}, # initially 5/4 * np.pi
                # # 'back': {'pos': (-0.4, -0.6, 1.72), 'ori': (7/8 * np.pi, 0, np.pi / 2)}  # (0, 3/4 * np.pi, 0)

                # #'top': {'pos': (0, -0.6, 1.8), 'ori': (np.pi, 0, 0)},
                # #'right': {'pos': (0, -0.4, 1.7), 'ori': (7 / 8 * np.pi, 0, 0)},  # initially 3/4 * np.pi
                # 'front': {'pos': (0.2, -0.4, 1.7), 'ori': (7 / 8 * np.pi, 0, -1)},  # (7 / 8 * np.pi, 0, -np.pi / 2)
                # 'left': {'pos': (0, -0.5, 1.7), 'ori': (-7 / 8 * np.pi, 0, 0)},  # initially 5/4 * np.pi
                # # 'back': {'pos': (-0.3, -0.55, 1.7), 'ori': (7/8 * np.pi, 0, np.pi / 2)}  # (0, 3/4 * np.pi, 0)

                #'top': {'pos': (0, -0.6, 1.8), 'ori': (np.pi, 0, 0)},
                # 'top': {'pos': (0.04, -0.55, 1.66), 'ori': (3.06, 0.01, -0.18)},

                # 'right': {'pos': (0, -0.4, 1.7), 'ori': (7 / 8 * np.pi, 0, 0)},  # initially 3/4 * np.pi

                # 'front': {'pos': (0.2, -0.4, 1.7), 'ori': (7 / 8 * np.pi, np.pi/4, 0)},  # (7 / 8 * np.pi, 0, -np.pi / 2)
                # 'front': {'pos': (0.15, -0.39, 1.7), 'ori': (2.84, 0.63, -0.1)},  # (7 / 8 * np.pi, 0, -np.pi / 2)

                #'left0': {'pos': (0, -0.7, 1.6), 'ori': (0, 0, 0)},
                #'left': {'pos': (0, -0.7, 1.6), 'ori': (np.pi, 0, 0)},  # initially 5/4 * np.pi
                #'left3': {'pos': (0, -0.7, 1.6), 'ori': (1 / 2 * np.pi, 0, 0)},
                #'left2': {'pos': (0, -0.58, 1.5), 'ori': (-2.83, 0.14, -0.01)},

                #'back': {'pos': (-0.2, -0.4, 1.7), 'ori': (7/8 * np.pi, 0, np.pi/2)}  # (0, 3/4 * np.pi, 0)
                #'back': {'pos': (-0.15, -0.39, 1.7), 'ori': (2.84, -0.63, -0.1)}
            }
            
            for key,item in object_views.items():
                print(f"Solve for {key}")
                pos = item['pos']
                ori = R.from_euler('xyz', item['ori']).as_quat()
                qu = inverse_kinematics.compute_target_configuration(pos, ori)
                # pybullet never fails it still returns some value
                qb = p.calculateInverseKinematics(sim.robot.id, sim.robot.ee_idx, pos, ori)[:-2]
                print("=====================")
                print("Our solution:  ", qu)
                print("From pybullet: ", qb)
                print("=====================")
                # exit()
            # exit()
            # pos = item['pos']
            # ori = R.from_euler('xyz', item['ori']).as_quat()
            # qu = inverse_kinematics.compute_target_configuration(pos, ori)
            # print(f"New Joint Configuration: {qu}")
            q  = qu
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
