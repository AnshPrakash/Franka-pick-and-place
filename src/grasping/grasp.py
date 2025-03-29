from abc import ABC, abstractmethod
import argparse

import pybullet as p

from scipy.spatial.transform import Rotation as R
import trimesh
import open3d as o3d
from open3d.visualization import draw_plotly
import open3d.core as o3c
import numpy as np

from giga.detection_implicit import gigaImplicit
from giga.grasp_sampler import GpgGraspSamplerPcl
from giga.utils import visual
from giga.utils.implicit import as_mesh



class Grasper(ABC):
    def __init__(self, obj_id=-1):
        """Base constructor for the Grasper class.
        The ground truth mesh file can already be loaded if object id is provided."""
        self.obj_id = obj_id
        self.obj_mesh = None
        if obj_id != -1:
            self.get_object_data(obj_id)

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

    @abstractmethod
    def get_grasps(self, obj_id, pose, best=True, visualize=False, include_gt=False):
        pass

    def execute_grasp(self):
        pass

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
    def __init__(self, model_path, obj_id=-1, quality_threshold=0.01): # 0.95
        super().__init__(obj_id)
        self.model_type = 'giga'
        self.model_path = model_path
        self.quality_threshold = quality_threshold
        self.grasp_generator = gigaImplicit(self.model_path, self.model_type, best=True, force_detection=True,
                                            qual_th=self.quality_threshold)

    def get_grasps(self, obj_id, pose, best=True, visualize=False, include_gt=False):
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
    def __init__(self, obj_id=-1, num_grasps=15, max_num_samples=100, num_parallel_workers=5, finger_depth=0.05, table_height=1.2): # table height is hardcoded based on visualization
        super().__init__(obj_id)
        self.num_grasps = num_grasps
        self.max_num_samples = max_num_samples
        self.num_parallel_workers = num_parallel_workers
        self.finger_depth = finger_depth
        self.table_height = table_height

        self.sampler = GpgGraspSamplerPcl(self.finger_depth-0.0075)

    def get_grasps(self, obj_id, pose, best=True, visualize=False, include_gt=False, ref_pc=None):
        """gt_pose for debugging: tuple with translation and rotation (as quaternion)"""
        # If the object id is different, load the new object data
        if obj_id != self.obj_id:
            self.obj_id = obj_id
            self.get_object_data(obj_id)
        # copy the gt mesh and transform it
        gt_mesh = o3d.geometry.TriangleMesh(self.obj_mesh)
        gt_mesh.transform(pose)
        pc = gt_mesh.sample_points_poisson_disk(number_of_points=10000, init_factor=5)

        grasps, grasps_pos, grasps_rot = self.sampler.sample_grasps_parallel(pc, num_parallel=self.num_parallel_workers,
                                                                        num_grasps=self.num_grasps, max_num_samples=150,
                                                                        safety_dis_above_table=self.table_height + self.finger_depth,
                                                                        show_final_grasps=False)
        if len(grasps) == 0:
            print("No grasps found")
            return None, None

        if len(grasps) > self.num_grasps:
            grasps = grasps[:self.num_grasps]
            grasps_pos = grasps_pos[:self.num_grasps]
            grasps_rot = grasps_rot[:self.num_grasps]

        if visualize:
            pc.paint_uniform_color([0, 0, 1])
            # for debugging merge the gt pointcloud
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
            if ref_pc is not None:
                ref_pc.paint_uniform_color([0, 1, 0])
                pc = pc + ref_pc

            self.visualize_grasps(grasps, pc)
        if best:
            return grasps[:1], None
        else:
            return grasps, None