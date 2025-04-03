import os
import glob

import pybullet as p
import yaml

import numpy as np

from typing import Dict, Any

from pybullet_object_models import ycb_objects  # type:ignore

from src.simulation import Simulation
from src.perception import Perception
from src.control import IKSolver
from src.moveIt import MoveIt
from src.grasping import ImplicitGrasper, SampleGrasper
from src.utils import extract_mesh_data, matrix_to_pose



def run_exp(config: Dict[str, Any]):
    # Example Experiment Runner File
    print("Simulation Start:")
    print(config['world_settings'], config['robot_settings'])
    object_root_path = ycb_objects.getDataPath()
    files = glob.glob(os.path.join(object_root_path, "Ycb*"))
    obj_names = [file.split('/')[-1] for file in files]
    print(obj_names)
    sim = Simulation(config)
    # PERCEPTION: initialize Perception class and load all necessary label meshes/pointclouds (atfer env has been reset the first time)
    perception = Perception(camera_stats=config['world_settings']['camera'])
    obstacle_init = True
    # GRASPING: initialize Grasper class
    #grasper = ImplicitGrasper('../../GIGA/data/models/giga_pile.pt')
    

    for obj_name in obj_names[-2:-1]:
        # PERCEPTION
        target_init = True
        for tstep in range(1):
            sim.reset(obj_name)
            print("Object name", obj_name)
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
            motion_controller = MoveIt(sim)
            perception.set_controller(motion_controller)

            # extract table height
            aabb_min, aabb_max = p.getAABB(sim.object.id)
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
            for i in range(80): # 50
                sim.step()

            # target_object_pcd = perception.get_pcd(sim.object.id, sim, use_static=False, use_ee=True, use_tsdf=False)
            # target_object_pos, failure = perception.perceive(sim.object.id, target_object_pcd, flatten=False, visualize=True)
            # target_object_pos, failure = perception.perceive(sim.object.id, target_object_pcd, flatten=False, visualize=False)
            # print(target_object_pos)

            #hammer 1.5
            # target_object_pos = np.array(
            #     [
            #         [ 0.99997107, -0.00670796,  0.00358629, -0.03395236],
            #         [ 0.0066931 ,  0.99996903,  0.00413986, -0.57064029],
            #         [-0.00361395, -0.00411574,  0.999985  ,  1.27670169],
            #         [ 0.        ,  0.        ,  0.        ,  1.        ]
            #     ]
            # )


            # Chips Can
            # target_object_pos = np.array(
            #     [[ 4.70818890e-01,  8.82229641e-01,  6.57911285e-04, -5.51576042e-02],
            #     [-8.82205258e-01 , 4.70811188e-01, -7.12095389e-03, -4.59120981e-01],
            #     [-6.59206859e-03 , 2.77226682e-03,  9.99974429e-01,  1.37937734e+00],
            #     [ 0.00000000e+00 , 0.00000000e+00,  0.00000000e+00,  1.00000000e+00]]
            # )

            #scissor 
            # target_object_pos = np.array(
            #     [[ 0.99887581, -0.02327664,  0.04129545, -0.09208208],
            #     [ 0.02407974 , 0.99952837 ,-0.01905793 ,-0.47589487],
            #     [-0.04083237 , 0.02003089 , 0.99896521 , 1.26693959],
            #     [ 0.         , 0.         , 0.         , 1.        ]]
            # )
            # target_obj_posi, target_obj_ori = matrix_to_pose(target_object_pos)
            # inverse_kinematics.debug("target Estimated pose", target_obj_posi, target_obj_ori)

            actual_position, actual_orientation = p.getBasePositionAndOrientation(sim.object.id)
            # inverse_kinematics.debug("target Actual pose", actual_position, actual_orientation)

            actual_pose_matrix = np.eye(4)
            from scipy.spatial.transform import Rotation as R  
            actual_pose_matrix[:3, :3] = R.from_quat(actual_orientation).as_matrix()  # Convert quaternion to rotation matrix
            actual_pose_matrix[:3, 3] = np.array(actual_position)

            #VISUALISE GRASP
            # target_object_pcd = perception.get_pcd(sim.object.id, sim, use_static=False, use_ee=True, use_tsdf=False)

            grasps, scores = grasper.get_grasps(sim.object.id, actual_pose_matrix, best=False, visualize=True, include_gt=True, ref_pc=None)
            vertices, triangles = extract_mesh_data(sim.object.id)
            inverse_kinematics.debug_object(sim.object.id, vertices, triangles)
            # for i, grasp in enumerate(grasps):
            #     possible_grasp = grasp.pose.as_matrix()
            #     grasp_position, grasp_orientation = matrix_to_pose(possible_grasp)
            #     inverse_kinematics.debug(f"Grasp pose {i} ", grasp_position, grasp_orientation)

            expected_grasp_pose = grasper.execute_grasp(actual_pose_matrix)

            # expected_grasp_pose = grasper.execute_grasp(target_object_pos)

            grasp_position, grasp_orientation = matrix_to_pose(expected_grasp_pose)

            # check grasp in RY config Faster than pybullet Rendering
            joint_states = sim.robot.get_joint_positions()
            inverse_kinematics.C.setJointState(joint_states)
            inverse_kinematics.debug("Grasp pose", grasp_position, grasp_orientation)

            goal_pos = sim.goal.goal_pos
            goal_pos = [goal_pos[0] - 0.1, goal_pos[1] - 0.1, goal_pos[2] +0.5 ] 
            goal_ori = R.from_euler('xyz', [np.pi, 0, 0]).as_quat()
            motion = MoveIt(sim)
            motion.moveTo(goal_pos, goal_ori)
            grasper.open_gripper()
            sim.close()
            exit()

            grasps, scores = grasper.get_grasps(sim.object.id, target_object_pos, best=False, visualize=True, include_gt=True, ref_pc=target_object_pcd)
            # print(f"Grasps: {grasps}")
            # print(f"Scores: {scores}")
            # grasp_pos = grasps[0].pose.translation
            # grasp_ori = grasps[0].pose.rotation.as_quat()
            # print(grasp_pos)
            # print(grasp_ori)

            # motion_controller.moveTo(grasp_pos, grasp_ori)
            #motion_controller.moveTo(grasp_pos, grasp_ori)

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
