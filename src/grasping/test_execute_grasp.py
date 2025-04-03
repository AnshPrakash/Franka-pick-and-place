import os
import glob

import pybullet
import yaml

import numpy as np

from typing import Dict, Any

from pybullet_object_models import ycb_objects  # type:ignore

from src.simulation import Simulation
from src.perception import Perception
from src.control import IKSolver
from src.moveIt import MoveIt
from src.grasping import ImplicitGrasper, SampleGrasper
from src.utils import visualize_point_cloud, get_robot_view_matrix, get_pcd_from_numpy


def run_exp(config: Dict[str, Any]):
    # Example Experiment Runner File
    print("Simulation Start:")
    print(config['world_settings'], config['robot_settings'])
    object_root_path = ycb_objects.getDataPath()
    files = glob.glob(os.path.join(object_root_path, "Ycb*"))
    obj_names = [file.split('/')[-1] for file in files]
    sim = Simulation(config)
    # PERCEPTION: initialize Perception class and load all necessary label meshes/pointclouds (atfer env has been reset the first time)
    perception = Perception(camera_stats=config['world_settings']['camera'])
    obstacle_init = True
    # GRASPING: initialize Grasper class
    #grasper = ImplicitGrasper('../../GIGA/data/models/giga_pile.pt')

    for obj_name in obj_names[2:3]:
        # PERCEPTION
        target_init = True
        for tstep in range(1):
            sim.reset(obj_name)
            grasper = SampleGrasper(sim)
            # PERCEPTION: only init obstacles once and target after object switch
            if obstacle_init:
                perception.set_objects([obst.id for obst in sim.obstacles])
                obstacle_init = False
            if target_init:
                perception.set_objects(sim.object.id)
                grasper.get_object_data(sim.object.id)
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

            inverse_kinematics = IKSolver(sim)
            # perception.set_ik_solver(inverse_kinematics)
            perception.set_controller(MoveIt(sim))

            # extract table height
            aabb_min, aabb_max = pybullet.getAABB(sim.object.id)
            print(aabb_min)
            print(aabb_max)
            # Convert to NumPy arrays for easier manipulation
            aabb_min = np.array(aabb_min)
            aabb_max = np.array(aabb_max)

            # Calculate dimensions: width (x-axis), depth (y-axis), height (z-axis)
            dimensions = aabb_max - aabb_min
            width, depth, height = dimensions

            print(f"Table Dimensions:\nWidth: {width}\nDepth: {depth}\nHeight: {height}")

            # ToDo: Check if initial skipping (due to falling object) is necessary
            for i in range(70): # 50
                sim.step()

            # target_object_pcd = perception.get_pcd(sim.object.id, sim, use_static=False, use_ee=True, use_tsdf=False)
            # target_object_pos, failure = perception.perceive(sim.object.id, target_object_pcd, flatten=False, visualize=True)
            

            target_object_pos = np.array([[ 0.98857895, -0.01786816,  0.14964086, -0.09125869],
                                           [ 0.0121794,   0.99917096,  0.03884666, -0.47388326],
                                           [-0.15021092, -0.03658046,  0.987977,    1.27830418],
                                           [ 0.,          0.,          0.,          1.        ]])
            print("target_object_pos", target_object_pos)

            expected_grasp_pose = grasper.execute_grasp(target_object_pos)
            

            from src.utils import matrix_to_pose
            final_position, final_ori = matrix_to_pose(expected_grasp_pose)
            robo_posi, robo_ori = sim.robot.get_ee_pose()
            
            print("Final position",final_position)
            print("Final Ori", final_ori)

            print("Robot position", robo_posi)
            print("Robot ori", robo_ori)

            inverse_kinematics.debug("Expected position of EE", final_position, final_ori)
            inverse_kinematics.debug("Actual position of EE", robo_posi, robo_ori)
            sim.close()
            exit()

            # grasps, scores = grasper.get_grasps(sim.object.id, target_object_pos, best=False, visualize=True, include_gt=True, ref_pc=target_object_pcd)
            grasps, scores = grasper.get_grasps(sim.object.id, target_object_pos, best=True, visualize=False)
            print(f"Grasps: {grasps}")
            print(f"Scores: {scores}")

            pos = target_object_pos[:3, 3]
            ori = target_object_pos[:3, :3]
            # convert rotation matrix to quaternion
            from scipy.spatial.transform import Rotation as R
            r = R.from_matrix(ori)
            quat = r.as_quat()
            target_object_pos = np.hstack([pos, quat])



            for i in range(10000):
                sim.step()
                # for getting renders
                #rgb, depth, seg = sim.get_ee_renders()
                #rgb, depth, seg = sim.get_static_renders()
                # PERCEPTION
                # obstacle_pcds = {obst.id: perception.get_pcd(obst.id, sim, use_ee=False) for obst in sim.obstacles}
                # obstacles_pos = np.array(
                #     [perception.perceive(obj_id, obj_pcd, visualize=False)[0] for (obj_id, obj_pcd) in obstacle_pcds.items()])
                # for obst in sim.obstacles:
                #     visualize_point_cloud(np.asarray(perception.object_pcds[obst.id].points))
                #visualize_point_cloud(np.asarray(target_object_pcd.points))

                obs_position_guess = np.zeros((2, 3))
                print((f"[{i}] Obstacle Position-Diff: "
                       f"{sim.check_obstacle_position(obs_position_guess)}"))
                goal_guess = np.zeros((7,))
                print((f"[{i}] Goal Obj Pos-Diff: "
                       f"{sim.check_goal_obj_pos(target_object_pos)}"))
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
