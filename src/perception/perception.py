"""Different perception algorithms, that can be leveraged by the agent"""
import copy

import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation as R
import trimesh
from src.utils import get_pcd_from_numpy, get_robot_view_matrix

import pybullet as p

class Perception:
    def __init__(self):
        self.object_meshs = {}
        self.object_pcds = {}
        # 0.05 in reference, but different scale
        # 0.005 won't filter much, but already not many points
        self.voxel_size = 0.005
        # if prior to registration already translate pointclouds to be roughly alined
        self.initial_translation = True
        # number of retries for registration in case exception happens or bad registration (no inliers)
        self.retries = 20

    def set_objects(self, object_ids, nb_points=10000):
        if not isinstance(object_ids, list):
            object_ids = [object_ids]
        for obj_id in object_ids:
            visual_data = p.getVisualShapeData(obj_id)
            # get global scaling of object
            scaling = visual_data[0][3]
            # passes the mesh file to load
            mesh_file = visual_data[0][4].decode('utf-8')
            # try with open3d and use trimesh if failed (due to non triangulated meshes)
            mesh = o3d.io.read_triangle_mesh(mesh_file, enable_post_processing=True)
            if mesh.is_empty():
                tri_mesh = trimesh.load(mesh_file)
                mesh = o3d.geometry.TriangleMesh(
                    o3d.utility.Vector3dVector(tri_mesh.vertices),
                    o3d.utility.Vector3iVector(tri_mesh.faces)
                )
            # scale along all axis
            mesh.vertices = o3d.utility.Vector3dVector(
                np.asarray(mesh.vertices) * np.array(scaling))
            # possibly recenter the mesh => https://www.open3d.org/docs/latest/tutorial/Basic/transformation.html might not be necessary?
            center = mesh.get_center()
            print(f"Mesh Center Object {obj_id}: ", center)
            mesh.translate(-center)
            print(f"After translating center for Object {obj_id}: ", mesh.get_center())
            self.object_meshs[obj_id] = mesh
            # results in better distributed sampling then uniform sampling
            pcd = mesh.sample_points_poisson_disk(number_of_points=nb_points, init_factor=5)
            #pcd = mesh.sample_points_uniformly(nb_points)
            self.object_pcds[obj_id] = pcd

    def preprocess_point_cloud(self, pcd):
        pcd_down = pcd.voxel_down_sample(self.voxel_size)

        radius_normal = self.voxel_size * 2
        pcd_down.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

        radius_feature = self.voxel_size * 5
        pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
            pcd_down,
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
        return pcd_down, pcd_fpfh

    def perceive(self, obj_id, pcd_target, fallback=np.eye(4), flatten=True, retries=None, logging=True, visualize=True):
        """Whole perception pipeline between source and target point clouds
        Args:
            obj_id: internal simulation id of object => to get the saved template point cloud
            pcd_target: detected point cloud from camera image
            fallback: Fallback transformation in case of failure
            flatten: Whether to output 4x4 transformation matrix or flattened 7d array (3d position, 4d orientation quaternion)
            retries: Number of retries for registration in case of failure (if not provided the self value is used)
            logging: Whether to print intermediate steps
            visualize: Whether to visualize pointclouds prior and after registration with open3d
            """
        # first step: get the source pcd of the object
        pcd_source = copy.deepcopy(self.object_pcds[obj_id])
        # translate source pcd to target pcd for more robust registration
        if self.initial_translation:
            initial_trans = pcd_target.get_center() - pcd_source.get_center()
            pcd_source.translate(initial_trans)

        # compute normals for local refinement
        pcd_source.estimate_normals()
        pcd_target.estimate_normals()

        # second step: preprocess the point clouds, meaning downsampling and feature extraction for RANSAC
        source_down, source_fpfh = self.preprocess_point_cloud(pcd_source)
        target_down, target_fpfh = self.preprocess_point_cloud(pcd_target)
        print(f"Source: {pcd_source}")
        print(f"Source Down: {source_down}")
        print(f"Target: {pcd_target}")
        print(f"Target Down: {target_down}")

        if visualize:
            self.draw_registration_result(source_down, target_down, np.identity(4))

        failure = False
        if retries is None:
            retries = self.retries

        if logging:
            print(f"Starting registration for object {obj_id}...")
        for i in range(retries):
            # third step: (fast) global registration using RANSAC
            try:
                result_global = self.global_registration(source_down, target_down, source_fpfh, target_fpfh)
                result_final = self.local_registration(pcd_source, pcd_target, result_global)
            except:
                print(f"Registration failed due to exception...")
                failure = True
                continue
            if result_final.fitness == 0:
                print(f"Registration failed due to no inliers...")
                failure = True
                continue
            else:
                print(f"Registration successful after {i+1} tries...")
                print(result_final)
                failure = False
                break

        if failure:
            trans_matrix = fallback
        else:
            trans_matrix = result_final.transformation.copy()

        if visualize:
            self.draw_registration_result(pcd_source, pcd_target, trans_matrix)

        # apply initial translation + additional check: in case we failed and have a fallback != identity: use this as it is
        if self.initial_translation and (not failure or np.array_equal(fallback, np.eye(4))):
            # with initial translation you have to account for rotation => the result transformation is with respective to the new coordinate system (the source pcd)
            initial_trans_matrix = np.identity(4)
            initial_trans_matrix[:3, 3] = initial_trans
            trans_matrix = trans_matrix @ initial_trans_matrix
        if flatten:
            pos = trans_matrix[:3, 3]
            ori = trans_matrix[:3, :3]
            # convert rotation matrix to quaternion
            r = R.from_matrix(ori)
            quat = r.as_quat()
            return np.hstack([pos, quat])
        else:
            return trans_matrix

    def draw_registration_result(self, source, target, transformation):
        import copy
        source_temp = copy.deepcopy(source)
        target_temp = copy.deepcopy(target)
        source_temp.paint_uniform_color([1, 0.706, 0])
        target_temp.paint_uniform_color([0, 0.651, 0.929])
        source_temp.transform(transformation)
        o3d.visualization.draw_geometries([source_temp, target_temp])

    def global_registration(self, source_down, target_down, source_features, target_features):
        """Fast global registration using RANSAC (based on open3d reference)"""
        # voxel_size * 0.5 in reference => way too small
        # 0.05 also not bad
        distance_threshold = 0.025
        result = o3d.pipelines.registration.registration_fgr_based_on_feature_matching(
            source_down, target_down, source_features, target_features,
            o3d.pipelines.registration.FastGlobalRegistrationOption(
                maximum_correspondence_distance=distance_threshold))
        return result

    def local_registration(self, source_pcd, target_pcd, result_global):
        """Local registration using ICP (based on open3d reference)"""
        # voxel_size * 0.4 in reference => way too small
        # 0.04 also not too bad
        distance_threshold = 0.02
        result = o3d.pipelines.registration.registration_icp(
            source_pcd, target_pcd, distance_threshold, result_global.transformation,
            o3d.pipelines.registration.TransformationEstimationPointToPlane())
        return result

    @staticmethod
    def pcd_from_img(width, height, depth_vals, view_matrix, proj_matrix, seg_mask=None, output_pcd_object=True):
        """
        Function to convert a camera image to a point cloud in world coordinates
        Based on: https://github.com/bulletphysics/bullet3/issues/1924
        Alternative: https://stackoverflow.com/questions/59128880/getting-world-coordinates-from-opengl-depth-buffer
        Args:
            width: width of image
            height: height of image
            depth_vals: Depth values of image, 1D array with length width*height
            view_matrix: view_matrix of camera
            proj_matrix: projection_matrix of camera
            seg_mask: Only consider part of the image depending on a segmentation mask
            output_pcd_object: Whether to output an open3d.geometry.PointCloud() object or a numpy array
            """
        # Set up the transformation matrix: from pixels+depth (in clip-space (NDC) as matrices operate on this space) to world coordinates.
        proj_matrix = np.asarray(proj_matrix).reshape([4, 4], order="F")
        view_matrix = np.asarray(view_matrix).reshape([4, 4], order="F")
        tran_pix_world = np.linalg.inv(np.matmul(proj_matrix, view_matrix))

        # Create a grid in NDC: x and y values range from -1 to 1
        # Note: This method directly creates a grid in NDC space
        # another approach works on the pixel positions: x = (2*w - img_width)/img_width, y = -(2*h - img_height)/img_height

        # => as in source, does not account for pixel center: y_grid, x_grid = np.mgrid[-1:1:2 / height, -1:1:2 / width]
        y, x = np.mgrid[-1 + 1 / height: 1: 2 / height, -1 + 1 / width: 1: 2 / width]
        #y, x = np.mgrid[-1:1:2 / height, -1:1:2 / width]
        # Flip y-axis to match NDC conventions (top of image -> y = 1)
        y *= -1.
        x, y, z = x.reshape(-1), y.reshape(-1), depth_vals.reshape(-1)

        # Flatten the segmentation mask (although should already be 1d) and select only those pixels that are True
        if seg_mask is not None:
            seg_indices = np.where(seg_mask.reshape(-1))
            x = x[seg_indices]
            y = y[seg_indices]
            z = z[seg_indices]

        # Create homogeneous coordinates
        h = np.ones_like(x)
        pixels = np.stack([x, y, z, h], axis=1)  # shape (N,4)
        # Filter out pixels with "infinite" depth (typically values near 1) => here all points > 0.99!
        #pixels = pixels[z < 0.99]
        # convert depth [0, 1] to NDC
        pixels[:, 2] = 2 * pixels[:, 2] -1

        # Transform these pixel coordinates from clip space to world space.
        points = np.matmul(tran_pix_world, pixels.T).T  # shape (N,4)
        # Do perspective division to obtain 3D points.
        points /= points[:, 3:4]
        points = points[:, :3]

        if output_pcd_object:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            return pcd
        else:
            return points

    @staticmethod
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
                id: Perception.pcd_from_img(sim.width, sim.height, depth, get_robot_view_matrix(pos, ori),
                                            sim.projection_matrix,
                                            seg == id, output_pcd_object=False) for id in object_ids}
        # merge pointclouds
        if use_ee:
            combined_dict = {key: get_pcd_from_numpy(np.vstack([static_obstacle_pcds[key], ee_pcds[key]])) for key in
                             object_ids}
        else:
            combined_dict = {key: get_pcd_from_numpy(static_obstacle_pcds[key]) for key in object_ids}

        return combined_dict