import os
import glob

import pybullet
import yaml

import numpy as np

from typing import Dict, Any

from pybullet_object_models import ycb_objects  # type:ignore

from src.simulation import Simulation
from src.perception import Perception
from src.utils import visualize_point_cloud, get_robot_view_matrix, get_pcd_from_numpy


def get_pcds(object_ids, sim, use_ee=True):
    """Extract pointclouds for the given objects based on both camera views."""
    if not isinstance(object_ids, list):
        object_ids = [object_ids]
    # first static
    rgb, depth, seg = sim.get_static_renders()
    static_obstacle_pcds = {
        id: Perception.pcd_from_img(sim.width, sim.height, depth, sim.stat_viewMat, sim.projection_matrix,
                                         seg == id, output_pcd_object=False) for id in object_ids}
    # then endeffector
    if use_ee:
        # ToDo: Logic to scan the room and combine pointclouds?
        rgb, depth, seg = sim.get_ee_renders()
        pos, ori = sim.robot.get_ee_pose()
        ee_pcds = {
            id: Perception.pcd_from_img(sim.width, sim.height, depth, get_robot_view_matrix(pos, ori), sim.projection_matrix,
                                        seg == id, output_pcd_object=False) for id in object_ids}
        print(ee_pcds)
    # merge pointclouds
    if use_ee:
        combined_dict = {key: get_pcd_from_numpy(np.vstack([static_obstacle_pcds[key], ee_pcds[key]])) for key in object_ids}
    else:
        combined_dict = {key: get_pcd_from_numpy(static_obstacle_pcds[key]) for key in object_ids}

    return combined_dict

def run_exp(config: Dict[str, Any]):
    # Example Experiment Runner File
    print("Simulation Start:")
    print(config['world_settings'], config['robot_settings'])
    object_root_path = ycb_objects.getDataPath()
    files = glob.glob(os.path.join(object_root_path, "Ycb*"))
    obj_names = [file.split('/')[-1] for file in files]
    sim = Simulation(config)
    # initialize Perception class and load all necessary label meshes/pointclouds (atfer env has been reset the first time)
    perception = Perception()
    obstacle_init = True
    target_init = True

    for obj_name in obj_names:
        target_init = True
        for tstep in range(10):
            sim.reset(obj_name)
            # only init obstacles once and target after object switch
            if obstacle_init:
                for obst in sim.obstacles:
                    visual_data = pybullet.getVisualShapeData(obst.id)
                    # passes the mesh file to load
                    mesh_file = visual_data[0][4].decode('utf-8')
                    perception.set_object(obst.id, mesh_file, scaling=visual_data[0][3])
                obstacle_init = False
            if target_init:
                visual_data = pybullet.getVisualShapeData(sim.object.id)
                mesh_file = visual_data[0][4].decode('utf-8')
                perception.set_object(sim.object.id, mesh_file, scaling=visual_data[0][3]) # hardcoded from simluation
                target_init = False
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

            # get pointcloud for target object once before

            target_object_pcd = get_pcds(sim.object.id, sim)[sim.object.id]
            target_object_pos = perception.perceive(sim.object.id, target_object_pcd, visualize=False)

            for i in range(10000):
                sim.step()
                # for getting renders
                #rgb, depth, seg = sim.get_ee_renders()
                #rgb, depth, seg = sim.get_static_renders()
                obstacle_pcds = get_pcds([obst.id for obst in sim.obstacles], sim)
                obstacles_pos = np.array(
                    [perception.perceive(obj_id, obj_pcd, visualize=False) for (obj_id, obj_pcd) in obstacle_pcds.items()])
                # for obst in sim.obstacles:
                #     visualize_point_cloud(np.asarray(perception.object_pcds[obst.id].points))
                #visualize_point_cloud(target_object_pcd)

                obs_position_guess = np.zeros((2, 3))
                print((f"[{i}] Obstacle Position-Diff: "
                       f"{sim.check_obstacle_position(obstacles_pos[:, :3])}"))
                goal_guess = np.zeros((7,))
                print((f"[{i}] Goal Obj Pos-Diff: "
                       f"{sim.check_goal_obj_pos(target_object_pos)}"))
                print(f"[{i}] Goal Satisfied: {sim.check_goal()}")
    sim.close()


if __name__ == "__main__":
    with open("configs/test_config.yaml", "r") as stream:
        try:
            config = yaml.safe_load(stream)
            print(config)
        except yaml.YAMLError as exc:
            print(exc)
    run_exp(config)
