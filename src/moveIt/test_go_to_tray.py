
import os
import glob
import yaml

import numpy as np
from scipy.spatial.transform import Rotation as R

from typing import Dict, Any

from pybullet_object_models import ycb_objects  # type:ignore

from src.simulation import Simulation
from src.moveIt import MoveIt


def run_exp(config: Dict[str, Any]):
    # Example Experiment Runner File
    print("Simulation Start:")
    print(config['world_settings'], config['robot_settings'])
    object_root_path = ycb_objects.getDataPath()
    files = glob.glob(os.path.join(object_root_path, "Ycb*"))
    obj_names = [file.split('/')[-1] for file in files]
    sim = Simulation(config, seed=13287)
    for obj_name in obj_names[0:1]:
        for tstep in range(1):
            sim.reset(obj_name)
            # print((f"Object: {obj_name}, Timestep: {tstep},"
            #        f" pose: {sim.get_ground_tuth_position_object}"))
            # pos, ori = sim.robot.pos, sim.robot.ori
            # print(f"Robot inital pos: {pos} orientation: {ori}")
            # l_lim, u_lim = sim.robot.lower_limits, sim.robot.upper_limits
            # print(f"Robot Joint Range {l_lim} -> {u_lim}")
            # sim.robot.print_joint_infos()
            # jpos = sim.robot.get_joint_positions()
            # print(f"Robot current Joint Positions: {jpos}")
            # jvel = sim.robot.get_joint_velocites()
            # print(f"Robot current Joint Velocites: {jvel}")
            # ee_pos, ee_ori = sim.robot.get_ee_pose()
            # print(f"Robot End Effector Position: {ee_pos}")
            # print(f"Robot End Effector Orientation: {ee_ori}")
            # goal_pos = sim.goal.goal_pos
            # goal_pos = [goal_pos[0] - 0.1, goal_pos[1] - 0.1, goal_pos[2] +0.5 ] 
            # goal_ori = R.from_euler('xyz', [np.pi, 0, 0]).as_quat()
            for i in range(300):
                sim.step()
            motion = MoveIt(sim)
            motion.go_to_tray()
            print("Yay! I am done with the task without collisions")
    sim.close()


if __name__ == "__main__":
    with open("../../configs/test_config.yaml", "r") as stream:
        try:
            config = yaml.safe_load(stream)
            print(config)
        except yaml.YAMLError as exc:
            print(exc)
    run_exp(config)











