from abc import ABC, abstractmethod
import argparse

import pybullet as p

from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
import trimesh
import open3d as o3d
from open3d.visualization import draw_plotly
import open3d.core as o3c
import numpy as np

from giga.detection_implicit import gigaImplicit
from giga.grasp_sampler import GpgGraspSamplerPcl
from giga.utils import visual
from giga.utils.implicit import as_mesh

from src.simulation import Simulation
from src.moveIt import MoveIt
from src.control import IKSolver
from src.utils import matrix_to_pose



class Grasper(ABC):
    def __init__(self, sim: Simulation, obj_id=-1, finger_depth=0.05, table_height=1.2):
        """Base constructor for the Grasper class.
        The ground truth mesh file can already be loaded if object id is provided."""
        self.obj_id = obj_id
        self.sim = sim
        self.obj_mesh = None
        self.motion_controller = MoveIt(sim)
        self.ik_solver = IKSolver(sim)
        if obj_id != -1:
            self.get_object_data(obj_id)

        self.table_height = table_height
        self.finger_depth = finger_depth

        # c = p.createConstraint(sim.robot.id,
        #                        9,
        #                        sim.robot.id,
        #                        10,
        #                        jointType=p.JOINT_GEAR,
        #                        jointAxis=[1, 0, 0],
        #                        parentFramePosition=[0, 0, 0],
        #                        childFramePosition=[0, 0, 0])
        # p.changeConstraint(c, gearRatio=-1, erp=0.5, maxForce=50)
        # p.changeDynamics(sim.robot.id, 9, lateralFriction=1.5)
        # p.changeDynamics(sim.robot.id, 10, lateralFriction=1.5)

    def get_object_data(self, obj_id):
        obj_data = p.getVisualShapeData(obj_id)
        mesh_file = obj_data[0][4].decode('utf-8')
        scaling = obj_data[0][3]
        mesh = o3d.io.read_triangle_mesh(mesh_file, enable_post_processing=True)
        mesh.vertices = o3d.utility.Vector3dVector(
            np.asarray(mesh.vertices) * np.array(scaling))
        center = mesh.get_center()
        print(f"Mesh Center Object {obj_id}: ", center)
        mesh.translate(-center)
        print(f"After translating center for Object {obj_id}: ", mesh.get_center())

        self.obj_mesh = mesh
        self.obj_id = obj_id

    @abstractmethod
    def get_grasps(self, obj_id, pose, best=True, visualize=False, include_gt=True, ref_pc=None,
                   debugging=False):
        pass
    
    def close_gripper(self):
        """
        Closes the Franka Panda gripper in the PyBullet simulation.
        
        This method assumes:
        - The Grasper instance has an attribute `robot` with:
            - robot.id: the PyBullet body ID for the robot,
            - robot.gripper_idx: a list of joint indices corresponding to the gripper.
        - A target joint position of 0.0 represents a fully closed gripper.
        
        It uses POSITION_CONTROL to command the gripper joints to the target position and
        steps the simulation to let the gripper actuate.
        """
        self.sim.robot.gripper_control([0.0, 0.0])
        # self.sim.robot.gripper_control([0.0, 0.0], forces=[1000, 1000])
        # Step the simulation a few hundred times to allow the gripper to close.
        for _ in range(25):
            self.sim.step()

        print("Gripper closed.")

    def open_gripper(self):
        """
        Opens the Franka Panda gripper in the PyBullet simulation.
        
        This method assumes:
        - The Grasper instance has an attribute `robot` with:
            - robot.id: the PyBullet body ID for the robot,
            - robot.gripper_idx: a list of joint indices corresponding to the gripper.
        - A target joint position of 0.4 represents a fully opened gripper.
        
        It uses POSITION_CONTROL to command the gripper joints to the target position and
        steps the simulation to let the gripper actuate.
        """
        
        self.sim.robot.gripper_control([0.1, 0.1])
        # Step the simulation a few hundred times to allow the gripper to close.
        for _ in range(25):
            self.sim.step()
        
        print("Gripper opened.")

    def execute_grasp(self, pose, obj_id=None, best=True, visualize=False, ref_pc=None, debugging=False):
        """
            Args: 
                pose
            Move the EE to the best grasp pose and then grip the object
        """
        if obj_id is None:
            obj_id = self.obj_id
        # Generate grasp candidates; assume get_grasps returns best grasp when best=True
        grasps, _ = self.get_grasps(obj_id, pose, best=best, visualize=visualize, ref_pc=ref_pc, debugging=debugging)
        if grasps is None or len(grasps) == 0:
            print("No valid grasp candidate found!")
            return None

        # try all grasps if solutions ar infeasible, but only max_retries if grasp did not get the object
        # => might be due to perception error, or previous grasp moved the object a little
        max_retries = 5

        for i, grasp in enumerate(grasps):
            try:
                final_grasp_pose = grasp.pose.as_matrix()  # A 4x4 homogeneous transformation matrix

                max_margin = 0.045
                z_threshold = self.table_height + self.finger_depth + 0.015 # heuristic for minimal grasp height
                approach_vector = final_grasp_pose[:3, 2]  # Extract the z-axis (approach vector)
                current_z = final_grasp_pose[2, 3]
                # final z value should not be below table height
                if approach_vector[2] < 0:
                    # Calculate the maximum allowed margin_distance such that the new z >= 1.2.
                    allowed_margin = (current_z - z_threshold) / (-approach_vector[2])
                    allowed_margin = max(0, allowed_margin)
                    # Use the lesser of the two values.
                    margin_distance = min(max_margin, allowed_margin)
                else:
                    margin_distance = max_margin

                final_grasp_pose[:3, 3] += margin_distance * approach_vector

                #DEBUGGGGGGGGG
                # # In case IK Solver fails during interpolation
                # final_position, final_ori = matrix_to_pose(final_grasp_pose)
                # fall_back_config = self.ik_solver.compute_target_configuration(final_position, final_ori)

                # self.motion_controller.moveTo(final_position, final_ori)
                # return final_grasp_pose
                #DEBUGGGGGGGG


                #print("Final grasp pose", final_grasp_pose)
                # final_grasp_pose  = np.array([  [  0.98896849 ,  0.12177069 , -0.08433993 , -0.13198916],
                #                                 [  0.11063837 , -0.98583892 , -0.12601893 , -0.5525772 ],
                #                                 [ -0.098491   ,  0.11529752 , -0.98843614 , 1.30148848],
                #                                 [  0.         ,  0.         ,  0.         ,  1.        ]])

                # Define a safe offset (in meters) for the pre-grasp pose along the grasp approach direction.
                safe_distance = 0.15

                # Assume that the approach direction is given by the third column (z-axis) of the rotation part.
                approach_vector = final_grasp_pose[:3, 2]
                # Compute pre-grasp pose by translating the final grasp pose backwards along the approach direction.
                pre_grasp_pose = final_grasp_pose.copy()
                pre_grasp_pose[:3, 3] -= safe_distance * approach_vector

                # --- Stage 1: Move to pre-grasp pose ---
                print("Moving to pre-grasp pose...")
                # This function is assumed to handle planning and execution.
                pre_grasp_position, pre_grasp_ori = matrix_to_pose(pre_grasp_pose)
                res = self.motion_controller.goTo(pre_grasp_position, pre_grasp_ori)
                if not res:
                    print(f"Pre-grasp pose {i+1} is not reachable, trying next grasp...")
                    continue
                # then open gripper
                self.open_gripper()

                # --- Stage 2: Linear approach ---
                print("Approaching final grasp pose from pre-grasp pose...")
                num_steps = 10

                # Extract rotations from the pre-grasp and final grasp poses.
                rot_pre = R.from_matrix(pre_grasp_pose[:3, :3])
                rot_final = R.from_matrix(final_grasp_pose[:3, :3])
                # Set up key times and create the Slerp object.
                key_times = np.array([0, 1])
                key_rots = R.from_quat(np.array([rot_pre.as_quat(), rot_final.as_quat()]))
                slerp_obj = Slerp(key_times, key_rots)

                # In case IK Solver fails during interpolation
                final_position, final_ori = matrix_to_pose(final_grasp_pose)
                fall_back_config = self.ik_solver.compute_target_configuration(final_position, final_ori)

                direct_try = False

                for step in range(1, num_steps + 1):
                    t = step / num_steps
                    # Linearly interpolate the translation between pre-grasp and final grasp.
                    interp_pos = (1 - t) * pre_grasp_pose[:3, 3] + t * final_grasp_pose[:3, 3]
                    # Spherical linear interpolation of the rotations using slerp.
                    interp_rot = slerp_obj([t])[0].as_matrix()

                    # Compose the full 4x4 transformation.
                    interp_pose = np.eye(4)
                    interp_pose[:3, :3] = interp_rot
                    interp_pose[:3, 3] = interp_pos

                    # Better fallback config:
                    q = self.ik_solver.compute_target_configuration(final_position, final_ori)
                    if q is not None:
                        fall_back_config = q

                    # Convert the matrix to a translation and quaternion using the helper function.
                    interim_position, interim_ori = matrix_to_pose(interp_pose)
                    q = self.ik_solver.compute_target_configuration(interim_position, interim_ori)
                    if q is None:
                        q = fall_back_config
                        direct_try = True


                    self.sim.robot.position_control(q)
                    ITER = 2 if step < num_steps and not direct_try else 100
                    for _ in range(ITER):
                        self.sim.step()  # Allow simulation to update each step

                    if direct_try:
                        break

                # --- Stage 3: Close the gripper ---
                print("Closing the gripper...")
                self.close_gripper()  # This function should command the gripper to close (see separate implementation).

                # --- Stage 4: Move to retreat grasp pose ---
                print("Moving to retreat grasp pose...")
                retreat_dis = 0.1

                # either move up in z-axis with respect to world frame (best for side grasps, but also in general)
                retreat_pose = final_grasp_pose.copy()
                retreat_pose[:3, 3] += np.array([0, 0, retreat_dis])
                # # EQUIVALENT TO
                # T = np.eye(4)
                # T[:3, 3] = np.array([0, 0, retreat_dis])
                # retreat_pose = np.dot(T, final_grasp_pose)

                # # or move in negative z-axis of grasps (so retreating in grasp direction) good for not side grasps
                # retreat_pose = final_grasp_pose.copy()
                # retreat_pose[:3, 3] -= retreat_dis * approach_vector
                # # # EQUIVALENT TO
                # # T = np.eye(4)
                # # T[:3, 3] = np.array([0, 0, -retreat_dis])
                # # retreat_pose = np.dot(final_grasp_pose, T)
                retreat_position, retreat_orientation = matrix_to_pose(retreat_pose)
                self.motion_controller.goTo(retreat_position, retreat_orientation)

                # simple check if grasp was successfull => check if gripper is fully closed (then no object inside)
                threshold = 0.0024
                gripper_state = self.sim.robot.get_gripper_positions()
                print(gripper_state)
                if gripper_state[0] < threshold and gripper_state[1] < threshold:
                    # try a few more grasps but eventually repeat perception
                    if i < max_retries:
                        continue
                    else:
                        return None
                else:
                    break
            except Exception as e:
                print(f"Grasp {i + 1} failed: {e}")
                continue

        print("Grasp executed successfully.")
        # return best_grasp
        return final_grasp_pose


    def visualize_grasps(self, grasps, pc):
        # Visualize point cloud and grasps
        grasps_scene = trimesh.Scene()
        grasp_mesh_list = [visual.grasp2mesh(g, score=1) for g in grasps]
        for i, g_mesh in enumerate(grasp_mesh_list):
            grasps_scene.add_geometry(g_mesh, node_name=f'grasp_{i}')
        # scene_mesh = get_scene_from_mesh_pose_list(mesh_pose_list, data_root=data_root)
        # composed_scene = trimesh.Scene([scene_mesh, grasps_scene])
        # composed_scene.show()
        mesh = as_mesh(grasps_scene)
        o3d_mesh = o3d.geometry.TriangleMesh(
            o3d.utility.Vector3dVector(mesh.vertices),
            o3d.utility.Vector3iVector(mesh.faces)
        )
        draw_plotly([pc,o3d_mesh])


class ImplicitGrasper(Grasper):
    def __init__(self, sim, model_path, obj_id=-1, quality_threshold=0.01): # 0.95
        super().__init__(sim, obj_id)
        self.model_type = 'giga'
        self.model_path = model_path
        self.quality_threshold = quality_threshold
        self.grasp_generator = gigaImplicit(self.model_path, self.model_type, best=True, force_detection=True,
                                            qual_th=self.quality_threshold)

    def get_grasps(self, obj_id, pose, best=True, visualize=False, include_gt=True, ref_pc=None,
                   debugging=False):
        """Generate grasp based on implicit learned model.
        Pose has to be a 4x4 transformation matrix."""
        # If the object id is different, load the new object data
        if obj_id != self.obj_id:
            self.obj_id = obj_id
            self.get_object_data(obj_id)
        # copy the gt mesh and transform it
        gt_mesh = o3d.geometry.TriangleMesh(self.obj_mesh)
        gt_mesh.transform(pose)

        # 1. First normalize to ensure the mesh fits in a unit cube (also used in the GIGA framework)

        # include some margin around the object
        margin_factor = 0.05
        bb = gt_mesh.get_axis_aligned_bounding_box()
        extents = bb.get_extent()
        margin = np.array(extents) * margin_factor

        # Expand the bounding box by the margin
        min_bound = bb.min_bound - margin
        max_bound = bb.max_bound + margin
        expanded_bb = o3d.geometry.AxisAlignedBoundingBox(min_bound, max_bound)

        center = expanded_bb.get_center()
        max_extent = np.max(expanded_bb.get_extent())

        # Translate the mesh so its center is at the origin
        gt_mesh.translate(-center)

        # Uniformly scale the mesh so that its maximum extent becomes 1.
        # (This maps the mesh roughly into the cube [-0.5, 0.5]^3)
        scale_factor = 1.0 / max_extent
        gt_mesh.scale(scale_factor, center=(0, 0, 0))

        # 2. generate the grid for the signed distance function also inputted into the network
        resolution = self.grasp_generator.resolution
        # ensure some constraints
        assert tuple(self.grasp_generator.pos.size()) == (1, resolution ** 3, 3)
        assert np.max(self.grasp_generator.pos.cpu().numpy()) <= 0.5
        grid_points = self.grasp_generator.pos.cpu().numpy().reshape(-1, 3) # internally saved grid

        # 3. compute the signed distance function

        tmesh = o3d.t.geometry.TriangleMesh.from_legacy(gt_mesh)

        # pc = tmesh.to_legacy().sample_points_poisson_disk(number_of_points=10000, init_factor=5)
        # o3d.visualization.draw_geometries([pc])

        scene = o3d.t.geometry.RaycastingScene()
        scene.add_triangles(tmesh)

        query_tensor = o3c.Tensor(grid_points, dtype=o3c.Dtype.Float32)
        sdf_values = scene.compute_signed_distance(query_tensor)

        # Reshape to (1, 40, 40, 40)
        tsdf_vol = sdf_values.reshape((1, resolution, resolution, resolution)).numpy().astype(np.float32)

        # 4. predict the grasp
        pc = gt_mesh.sample_points_poisson_disk(number_of_points=10000, init_factor=5) # in tutorial also pointcloud was inputted
        o3d.visualization.draw_geometries([pc])
        grasps, scores, timings = self.grasp_generator(state=argparse.Namespace(tsdf=tsdf_vol, pc=pc))
        print(f"Grasps: {grasps}")
        print(f"Scores: {scores}")
        if len(grasps) == 0:
            return None, None

        # first visualize in the normalized space
        if visualize:
            self.visualize_grasps(grasps, pc)

        # 5. Denormalize the grasps
        pc.scale(1.0 / scale_factor, center=(0, 0, 0))
        pc.translate(center)
        for grasp in grasps:
            grasp.pose.translation = grasp.pose.translation / scale_factor + center # rotation remains unchanged
            grasp.width = grasp.width / scale_factor

        # visualize again for comparison
        if include_gt is not None:
            gt_pose = p.getBasePositionAndOrientation(obj_id)
            translation = gt_pose[0]
            rotation_matrix = R.from_quat(gt_pose[1]).as_matrix()

            trans_matrix = np.eye(4)
            trans_matrix[:3, :3] = rotation_matrix
            trans_matrix[:3, 3] = translation

            gt_mesh = o3d.geometry.TriangleMesh(self.obj_mesh)
            gt_mesh.transform(trans_matrix)
            gt_pc = gt_mesh.sample_points_poisson_disk(number_of_points=10000, init_factor=5)

            gt_pc.paint_uniform_color([1, 0, 0])

            pc = pc + gt_pc
        if visualize:
            self.visualize_grasps(grasps, pc)

        if best:
            max_idx = np.argmax(scores)
            return grasps[max_idx:max_idx+1], scores[max_idx:max_idx+1]
        else:
            return grasps, scores

class SampleGrasper(Grasper):
    def __init__(self, sim, obj_id=-1, num_grasps=15, max_num_samples=100, num_parallel_workers=5, finger_depth=0.05, table_height=1.2):
        super().__init__(sim, obj_id, finger_depth, table_height)
        self.num_grasps = num_grasps
        self.max_num_samples = max_num_samples
        self.num_parallel_workers = num_parallel_workers
        self.finger_depth = finger_depth

        self.sampler = GpgGraspSamplerPcl(self.finger_depth-0.0075)

    def get_grasps(self, obj_id, pose, best=True, visualize=False, include_gt=True, ref_pc=None,
                   debugging=False):
        """gt_pose for debugging: tuple with translation and rotation (as quaternion)
        
        Args:
            obj_id: Internal simulation id of the object to retrieve the saved template point cloud.
            pose: A 4x4 transformation matrix to transform the object's mesh.
            best: If True, return only one grasp candidate (the best one by default).
            visualize: Whether to visualize the point cloud with the candidate grasps.
            include_gt: Whether to include ground-truth point cloud in the visualization.
            ref_pc: Reference point cloud to use in the visualization.
        
        Returns:
            If best is True, returns a list containing one grasp candidate (either the best one or a random one)
            and None for scores; otherwise, returns all candidates and None.
        """
        # If the object id is different, load the new object data
        if obj_id != self.obj_id:
            self.obj_id = obj_id
            self.get_object_data(obj_id)
        
        # Copy the ground truth mesh and transform it
        gt_mesh = o3d.geometry.TriangleMesh(self.obj_mesh)
        gt_mesh.transform(pose)
        pc = gt_mesh.sample_points_poisson_disk(number_of_points=10000, init_factor=5)
        
        # Sample grasp candidates from the point cloud using the grasp sampler.
        grasps, grasps_pos, grasps_rot = self.sampler.sample_grasps_parallel(
            pc,
            num_parallel=self.num_parallel_workers,
            num_grasps=self.num_grasps,
            max_num_samples=175,
            safety_dis_above_table= self.table_height + 0.005, # 0.005 also used in sample code
            show_final_grasps=False
        )
        if len(grasps) == 0:
            print("No grasps found")
            return None, None

        # Trim candidate lists to the specified number of grasps.
        if len(grasps) > self.num_grasps:
            grasps = grasps[:self.num_grasps]
            grasps_pos = grasps_pos[:self.num_grasps]
            grasps_rot = grasps_rot[:self.num_grasps]
        
        # Visualization (optional)
        if visualize:
            pc.paint_uniform_color([0, 0, 1]) #blue
            if debugging:
                p.addUserDebugPoints(
                    pointPositions=pc.points,
                    pointColorsRGB=[[0, 0, 1]] * len(pc.points),
                    pointSize=4,
                )
            # for debugging merge the gt pointcloud
            if include_gt:
                gt_pose = p.getBasePositionAndOrientation(obj_id)
                translation = gt_pose[0]
                rotation_matrix = R.from_quat(gt_pose[1]).as_matrix()
                trans_matrix = np.eye(4)
                trans_matrix[:3, :3] = rotation_matrix
                trans_matrix[:3, 3] = translation
                gt_mesh = o3d.geometry.TriangleMesh(self.obj_mesh)
                gt_mesh.transform(trans_matrix)
                gt_pc = gt_mesh.sample_points_poisson_disk(number_of_points=10000, init_factor=5)
                gt_pc.paint_uniform_color([1, 0, 0]) # red
                if debugging:
                    p.addUserDebugPoints(
                        pointPositions=gt_pc.points,
                        pointColorsRGB=[[1, 0, 0]] * len(gt_pc.points),
                        pointSize=4,
                    )
                pc = pc + gt_pc
            if ref_pc is not None:
                ref_pc.paint_uniform_color([0, 1, 0]) # Green
                if debugging:
                    p.addUserDebugPoints(
                        pointPositions=ref_pc.points,
                        pointColorsRGB=[[0, 1, 0]] * len(ref_pc.points),
                        pointSize=4,
                    )
                pc = pc + ref_pc
            self.visualize_grasps(grasps, pc)
            o3d.visualization.draw_geometries([pc])
        
        # Return candidate(s)
        if best:
            # Todo: Implement a better scoring function
            ordered_grasps = sorted(grasps, key=lambda g: self.score_grasp(g), reverse=True)
            best_grasp = grasps[:1]
            #self.visualize_grasps(best_grasp, pc)
            return ordered_grasps, None
        else:
            import random
            chosen = random.choice(grasps)
            return [chosen], None

    def score_grasp(self, grasp):
        """Naive scoring: All grasps are already good and valid => choose the one that is most aligned with the world negative z-axis"""
        ee_z = grasp.pose.as_matrix()[:3, 2]
        ee_z = ee_z / np.linalg.norm(ee_z)
        target_direction = np.array([0, 0, -1])
        return np.dot(ee_z, target_direction)
