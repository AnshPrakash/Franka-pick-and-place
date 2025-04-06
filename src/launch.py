import argparse
import os
import glob
import yaml

import numpy as np
from scipy.spatial.transform import Rotation as R

from typing import Dict, Any

from pybullet_object_models import ycb_objects  # type:ignore

from src.simulation import Simulation
from src.moveIt import MoveIt
from src.simulation import Simulation
from src.perception import Perception
from src.grasping import SampleGrasper



def run_experiment(args: argparse.ArgumentParser, config: Dict):
    # Example experiment function: replace this with your actual experiment code.
    print("Starting experiment with the following settings:")
    print(f"Ycb object  {args.object}")
    print(f"Obstacles are {args.obstacles}")
    print(f"Recording: {args.record}")
    print("Simulation Start:")
    print(config['world_settings'], config['robot_settings'])
    object_root_path = ycb_objects.getDataPath()
    files = glob.glob(os.path.join(object_root_path, "Ycb*"))
    obj_names = [file.split('/')[-1] for file in files]
    ycb_object = args.object
    if ycb_object not in obj_names:
        print("Wrong YcbObject")
    sim = Simulation(config, seed=42)

    # Set the object
    sim.reset(ycb_object)


    # Let the object fall and stabalise
    for i in range(200):
        sim.step()

    # Setup Perception module
    # PERCEPTION: initialize Perception class and load all necessary label meshes/pointclouds (atfer env has been reset the first time)
    perception = Perception(camera_stats=config['world_settings']['camera'])
    obstacle_init = True
    target_init = True
    if obstacle_init:
        perception.set_objects([obst.id for obst in sim.obstacles])
        obstacle_init = False
    if target_init:
        perception.set_objects(sim.object.id)
        target_init = False

    motion_controller = MoveIt(sim)
    perception.set_controller(motion_controller)

    # Perception Stage
    target_object_pcd = perception.get_pcd(sim.object.id, sim, use_static=False, use_ee=True, use_tsdf=False)
    target_object_pos, failure = perception.perceive(sim.object.id, target_object_pcd, visualize=False)
    

    # Grasping Stage
    grasper = SampleGrasper(sim)
    iter = 0
    expected_grasp_pose = None
    while expected_grasp_pose is None and iter < 5:
        target_object_pcd = perception.get_pcd(sim.object.id, sim, use_static=False, use_ee=True, use_tsdf=False)
        target_object_pos, failure = perception.perceive(sim.object.id, target_object_pcd, flatten=False,
                                                            visualize=False)
        expected_grasp_pose = grasper.execute_grasp(target_object_pos, best=True, visualize=False, ref_pc=target_object_pcd, debugging=False)

        iter += 1
    

    # Path planning towards goal with obstacle avoidance
    motion_controller.go_to_tray()
    grasper.open_gripper()

    print("Yay! We did it")
    sim.close()
    

if __name__ == "__main__":
    # Set up the argument parser
    parser = argparse.ArgumentParser(description="Run an experiment with specified parameters.")
    parser.add_argument('--object', type=str, default="YcbCrackerBox", help='YcbObject name')
    parser.add_argument('--obstacles', type=bool, default=True, help='Include obstacles in the simulation')
    parser.add_argument('--record', type=bool, default=False, help='Record the simulation')
    
    # Parse the arguments from the command line
    args = parser.parse_args()

    with open("../configs/test_config.yaml", "r") as stream:
        try:
            config = yaml.safe_load(stream)
            print(config)
        except yaml.YAMLError as exc:
            print(exc)
    with open("../configs/custom_config.yaml", "r") as stream:
        try:
            custom_config = yaml.safe_load(stream)
            print(custom_config)
        except yaml.YAMLError as exc:
            print(exc)
    config.update(custom_config)
    
    run_experiment(args, config)

