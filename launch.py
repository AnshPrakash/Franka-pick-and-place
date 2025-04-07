import argparse
import os
import glob
import yaml
import json

import numpy as np
from scipy.spatial.transform import Rotation as R

from typing import Dict, Any
import pybullet as p

from pybullet_object_models import ycb_objects  # type:ignore

from src.simulation import Simulation
from src.moveIt import MoveIt
from src.simulation import Simulation
from src.perception import Perception
from src.grasping import SampleGrasper
import gc


def run_experiment(args: argparse.ArgumentParser, config: Dict):
    print("Starting experiment with the following settings:")
    print(config['world_settings'], config['robot_settings'])
    print(f"Obstacles are {args.obstacles}")
    print(f"Evaluation steps are {args.steps}")
    print(f"Recording: {args.record}")
    object_root_path = ycb_objects.getDataPath()
    files = glob.glob(os.path.join(object_root_path, "Ycb*"))
    obj_names = [file.split('/')[-1] for file in files]
    if not args.all:
        ycb_object = args.object
        if ycb_object not in obj_names:
            raise ValueError("Wrong YcbObject")
        obj_names = [args.object]
    recording = args.record

    allow_obstacles = args.obstacles
    config["world_settings"]["turn_on_obstacles"] = allow_obstacles

    print("Simulation Start:")
    sim = Simulation(config, seed=42)
    perception = Perception(camera_stats=config['world_settings']['camera'])
    obstacle_init = True

    # objects for evaluation
    results = {}
    exceptions = {}
    try:
        if recording:
            # Check current connection info.
            conn_info = p.getConnectionInfo()
            if conn_info.get("connectionMethod", None) != p.GUI:
                print("Recording requires GUI mode. Reconnecting in GUI mode...")
                p.disconnect()
                p.connect(p.GUI)
            print("Start recording...")
            log_id = p.startStateLogging(p.STATE_LOGGING_VIDEO_MP4, "simulation_recording.mp4")
        for ycb_object in obj_names:
            # provide an entry for every object
            results[ycb_object] = {"attempts": 0, "grasp_success": 0, "goal_success": 0}
            exceptions[ycb_object] = 0

            target_init = True

            for tstep in range(args.steps):
                grasp_success = False
                goal_success = False
                exception_occurred = False
                try:
                    # Set the object
                    sim.reset(ycb_object)
                    print("Object name", ycb_object)

                    # initialize or configure task objects
                    grasper = SampleGrasper(sim)
                    motion_controller = MoveIt(sim)
                    perception.set_controller(motion_controller)

                    if obstacle_init:
                        perception.set_objects([obst.id for obst in sim.obstacles])
                        obstacle_init = False
                    if target_init:
                        perception.set_objects(sim.object.id)
                        grasper.get_object_data(sim.object.id)
                        target_init = False

                    # Let the object fall and stabalise => sometimes falling for a long time
                    for i in range(250):
                        sim.step()

                    #### Perception & Grasping Stage ####
                    iter = 0
                    expected_grasp_pose = None
                    while expected_grasp_pose is None and iter < 5:
                        target_object_pcd = perception.get_pcd(sim.object.id, sim, use_static=False, use_ee=True, use_tsdf=False)
                        target_object_pos, failure = perception.perceive(sim.object.id, target_object_pcd, flatten=False,
                                                                         visualize=False)
                        expected_grasp_pose = grasper.execute_grasp(target_object_pos, obj_id=sim.object.id, best=True, visualize=False,
                                                                    ref_pc=target_object_pcd, debugging=False)

                        iter += 1

                    # check if grasping was successful
                    if expected_grasp_pose is not None:
                        grasp_success = True

                    if grasp_success:
                        # Path planning towards goal with obstacle avoidance
                        motion_controller.go_to_tray()
                        grasper.open_gripper()

                        # wait a few steps to let the object fall into the tray
                        for i in range(50):
                            sim.step()

                        # check if goal reached
                        if sim.check_goal():
                            goal_success = True

                        # check twice => in case of obstacles: not that goal is reached but no collision occurs
                        for i in range(50):
                            sim.step()

                        # check if goal reached
                        if sim.check_goal():
                            goal_success = True

                    gc.collect()
                except Exception as e:
                    print(f"Exception occurred: {e}")
                    print("______________________________________\n\n")

                    exception_occurred = True

                    try:
                        sim.close()
                    except Exception as e:
                        pass
                    gc.collect()
                    # use pre-initialization steps of simulation
                    # starting global_settings
                    p.connect(sim.gui_mode)
                    p.setTimeStep(sim.timestep)
                    import pybullet_data
                    p.setAdditionalSearchPath(pybullet_data.getDataPath())
                finally:
                    # update all counts at the end
                    print(f"Grasp Success: {grasp_success}, Goal Success: {goal_success}, Exception: {exception_occurred}")
                    results[ycb_object]["attempts"] += 1
                    if grasp_success:
                        results[ycb_object]["grasp_success"] += 1
                    if goal_success:
                        results[ycb_object]["goal_success"] += 1
                    if exception_occurred:
                        exceptions[ycb_object] += 1

            # write all metrics to a json file
            # Log the success metrics to a JSON file.
            post_string = "_obstacles" if allow_obstacles else ""
            if args.all:
                post_string = "_all" + post_string
            else:
                post_string = "_" + args.object + post_string
            post_string += "_" + str(args.steps)
            with open(f"success_results{post_string}.json", "w") as f:
                json.dump(results, f, indent=4)

            # Log the exception counts to a separate JSON file.
            with open(f"exceptions_results{post_string}.json", "w") as f:
                json.dump(exceptions, f, indent=4)
    finally:
        if recording:
            # Stop recording
            p.stopStateLogging(log_id)
            print("Stop recording")
        print("Close simulation")
        sim.close()
    

if __name__ == "__main__":
    # Set up the argument parser
    parser = argparse.ArgumentParser(description="Run an experiment with specified parameters.")
    parser.add_argument('--object', type=str, default="YcbCrackerBox", help='YcbObject name')
    parser.add_argument('--all', action='store_true', help='Whether all objects should be evaluated or not')
    parser.add_argument('--steps', type=int, default=10, help='Evaluation iterations per object')
    parser.add_argument('--obstacles', action='store_true', help='Include obstacles in the simulation')
    parser.add_argument('--record', type=bool, default=False, help='Record the simulation')
    
    # Parse the arguments from the command line
    args = parser.parse_args()

    with open("configs/test_config.yaml", "r") as stream:
        try:
            config = yaml.safe_load(stream)
            print(config)
        except yaml.YAMLError as exc:
            print(exc)
    with open("configs/custom_config.yaml", "r") as stream:
        try:
            custom_config = yaml.safe_load(stream)
            print(custom_config)
        except yaml.YAMLError as exc:
            print(exc)
    config.update(custom_config)

    run_experiment(args, config)

