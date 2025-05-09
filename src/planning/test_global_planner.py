import os
import glob
import yaml

import numpy as np
from scipy.spatial.transform import Rotation as R

from typing import Dict, Any

from pybullet_object_models import ycb_objects  # type:ignore

from src.simulation import Simulation
from src.planning import Global_planner


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
            gp = Global_planner(sim)
            goal_pos = sim.goal.goal_pos
            goal_pos = [goal_pos[0] - 0.1, goal_pos[1] - 0.1, goal_pos[2] +0.5 ] 
            goal_ori = R.from_euler('xyz', [np.pi, 0, 0]).as_quat()
            path = gp.plan(goal_pos, goal_ori)
            # print(path)
            replan_freq = 10
            init_config = sim.robot.get_joint_positions()
            print("Initial Robot Config:", init_config)
            if len(path) > 10:
                q = path[10]
            else:
                q = path[1]
            sim.robot.position_control(q)
            for i in range(10000):    
                sim.step()
                if i%replan_freq == 0:
                    print("Initial Config: ", init_config)
                    print("Updated Robot Config:", sim.robot.get_joint_positions())
                    path = gp.plan(goal_pos, goal_ori)
                # sim.robot.position_control(path[-1])
                # for _ in range(100):
                #     sim.step()
                # print("Updated Robot Config:", sim.robot.get_joint_positions())
                    if len(path) > 10:
                        q = path[10]
                    else:
                        q = path[1]
                    
                    sim.robot.position_control(q)
                    for _ in range(2):
                        sim.step()
                # for q in path:
                #     # Update the robot joint positions in simulation (use appropriate API)
                #     sim.robot.position_control(q)
                #     # Optionally, update the configuration view if needed:
                #     # gp.C.setJointState(q)  # if you want to see the updated configuration in RAi viewer
                #     # Step the simulation for a few iterations for smooth execution.
                #     # too slow with this line
                #     for _ in range(10):
                #         sim.step()
                # for getting renders
                # rgb, depth, seg = sim.get_ee_renders()
                # rgb, depth, seg = sim.get_static_renders()
                # obs_position_guess = np.zeros((2, 3))
                # print((f"[{i}] Obstacle Position-Diff: "
                #        f"{sim.check_obstacle_position(obs_position_guess)}"))
                # goal_guess = np.zeros((7,))
                # print((f"[{i}] Goal Obj Pos-Diff: "
                #        f"{sim.check_goal_obj_pos(goal_guess)}"))
                # print(f"[{i}] Goal Satisfied: {sim.check_goal()}")
    sim.close()


if __name__ == "__main__":
    with open("../../configs/test_config.yaml", "r") as stream:
        try:
            config = yaml.safe_load(stream)
            print(config)
        except yaml.YAMLError as exc:
            print(exc)
    run_exp(config)
