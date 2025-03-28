"""Different perception algorithms, that can be leveraged by the agent"""
import copy

import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation as R
import trimesh
from src.utils import get_pcd_from_numpy, get_robot_view_matrix

import pybullet as p

class Perception:
    def __init__(self, voxel_size=0.005, retries=42, initial_translation=True,
                 camera_stats=None): # extra parameters for TSDF method
        self.object_meshs = {}
        self.object_pcds = {}
        # 0.05 in reference, but different scale
        # 0.005 won't filter much, but already not many points
        self.voxel_size = voxel_size
        # if prior to registration already translate pointclouds to be roughly alined
        self.initial_translation = initial_translation
        # number of retries for registration in case exception happens or bad registration (no inliers)
        self.retries = retries

        self.object_views = {
            'top': {'pos': (0, -0.65, 1.9), 'ori': (np.pi, 0, 0)},
            'right': {'pos': (0, -0.3, 1.9), 'ori': (7/8 * np.pi, 0, 0)}, # initially 3/4 * np.pi
            'left': {'pos': (0, -0.9, 1.9), 'ori': (9/8 * np.pi, 0, 0)}, # initially 5/4 * np.pi
            'front': {'pos': (0.4, -0.65, 1.9), 'ori': (7/8 * np.pi, 0, -np.pi / 2)}, # (0, -3/4 * np.pi, 0)
            'back': {'pos': (-0.4, -0.65, 1.9), 'ori': (7/8 * np.pi, 0, np.pi / 2)}  # (0, 3/4 * np.pi, 0)

        }

        self.ik_solver = None

        self.camera_stats = camera_stats

    def set_ik_solver(self, ik_solver):
        self.ik_solver = ik_solver

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
        if not isinstance(obj_id, o3d.geometry.PointCloud):
            pcd_source = copy.deepcopy(self.object_pcds[obj_id])
        else:
            pcd_source = copy.deepcopy(obj_id)
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
                result_final = self.local_registration(pcd_source, pcd_target, result_global.transformation)
            except:
                print(f"Registration failed due to exception...")
                failure = True
                continue
            if result_final.fitness == 0:
                # print(f"Registration failed due to no inliers...")
                failure = True
                continue
            else:
                print(f"Registration successful after {i+1} tries...")
                print(result_final)
                failure = False
                break
        # try 1 time with point to point
        if failure:
            for i in range(5): # also 5 tries to be safe, but usually only 1 try needed
                result_global = self.global_registration(source_down, target_down, source_fpfh, target_fpfh)
                result_final = self.local_registration(pcd_source, pcd_target, result_global.transformation, point_to_plane=False)
                if result_final.fitness != 0:
                    print(f"Registration successful with PointToPoint...")
                    failure = False
                    break
        if failure:
            print(f"Registration failed after {retries} retries...")
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
            return np.hstack([pos, quat]), failure
        else:
            return trans_matrix, failure

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

    def local_registration(self, source_pcd, target_pcd, global_transformation, point_to_plane=True):
        """Local registration using ICP (based on open3d reference)"""
        # voxel_size * 0.4 in reference => way too small
        # 0.04 also not too bad
        distance_threshold = 0.02
        # Note: originally used PointToPlane ICP, but PointToPoint seems to work better
        if point_to_plane:
            result = o3d.pipelines.registration.registration_icp(
                source_pcd, target_pcd, distance_threshold, global_transformation,
                o3d.pipelines.registration.TransformationEstimationPointToPlane())
        else:
            result = o3d.pipelines.registration.registration_icp(
                source_pcd, target_pcd, distance_threshold, global_transformation,
                o3d.pipelines.registration.TransformationEstimationPointToPoint())
        return result


    def get_pcd(self, object_id, sim, use_static=True, use_ee=False, use_tsdf=False):
        """Extract pointclouds for the given object based on both camera views."""
        # point cloud collection for later stacking
        pcd_collections = []
        # first static
        if use_static:
            rgb, depth, seg = sim.get_static_renders()
            static_pcd = Perception.pcd_from_img(
                sim.width,
                sim.height,
                depth,
                sim.stat_viewMat,
                sim.projection_matrix,
                seg == object_id,
                output_pcd_object=False
            )
            pcd_collections.append(static_pcd)
        # then endeffector
        if use_ee:
            # pos_threshold = 0.19
            # ori_threshold = 0.19
            scene_data = []
            # NOTE!!!: for a lower threshold (0.004) the sim.step() function is called very often and causes the simulation
            # in the next object iteration to crash due to lost physics server connection (just limiting steps to 25 also did work)
            change_threshold = 0.01
            stored_trans = self.initial_translation
            self.initial_translation = False
            for view, view_params in self.object_views.items():
                target_pos = view_params['pos']
                target_ori = R.from_euler('xyz', view_params['ori']).as_quat()

                # move robot ee to desired positions
                # ToDo: change with own IK solver!
                #q = self.ik_solver.compute_target_configuration(target_pos, target_ori)
                q = p.calculateInverseKinematics(sim.robot.id, sim.robot.ee_idx, target_pos, target_ori)[:-2]
                # if solution is not found /infeasible just skip
                if q is None:
                    print("View skipped due to infeasible IK solution...")
                    continue
                sim.robot.position_control(q)
                sim.step()
                last_pos, last_ori = target_pos, target_ori
                curr_pos, curr_ori = sim.robot.get_ee_pose()
                # adjust ee until position and orientation change below some threshold
                while (np.linalg.norm(curr_pos - last_pos) > change_threshold or
                       np.linalg.norm(curr_ori - last_ori) > change_threshold):
                    sim.robot.position_control(q)
                    sim.step()
                    last_pos, last_ori = curr_pos, curr_ori
                    curr_pos, curr_ori = sim.robot.get_ee_pose()
                    # print(f"View {view}, Pos change:", np.linalg.norm(curr_pos - last_pos))
                    # print(f"View {view}, Ori change:", np.linalg.norm(curr_ori - last_ori))
                print(f"View {view}, Final Pos error:", np.linalg.norm(sim.robot.get_ee_pose()[0] - target_pos))
                print(f"View {view}, Final Ori error", np.linalg.norm(sim.robot.get_ee_pose()[1] - target_ori))

                # get the new depth image and add pointcloud
                rgb, depth, seg = sim.get_ee_renders()
                pos, ori = sim.robot.get_ee_pose()

                # append current scene information
                scene_data.append({'color': rgb, 'depth': depth, 'view_matrix': get_robot_view_matrix(pos, ori), 'seg': seg})

            # compute pointclouds from scene data
            if use_tsdf:
                ee_pcd = self.integrate_tsdf(scene_data, sim.width, sim.height,
                                             self.camera_stats['near'], self.camera_stats['far'], self.camera_stats['fov'])
                pcd_collections.append(ee_pcd.points)
            else:
                ee_pcds = []
                for data in scene_data:
                    ee_pcd = Perception.pcd_from_img(
                        sim.width,
                        sim.height,
                        data['depth'],
                        data['view_matrix'],
                        sim.projection_matrix,
                        data['seg'] == object_id,
                        output_pcd_object=False
                    )
                    ee_pcd_object = get_pcd_from_numpy(ee_pcd)
                    ee_pcd_object.estimate_normals()
                    ee_pcds.append((len(ee_pcd_object.points), ee_pcd_object))
                # align pointclouds
                ee_pcds.sort(key=lambda x: x[0], reverse=True)
                reference_pcd = ee_pcds[0][1] # point cloud with most points is reference
                pcd_collections.append(reference_pcd.points)
                for nb_points, pcd in ee_pcds[1:]:
                    if nb_points > 0:
                        # current pcd is source as we want to transform it to the reference
                        #trans = self.local_registration(ee_pcd_object, reference_pcd, np.eye(4)).transformation
                        trans, failure = self.perceive(pcd, reference_pcd, flatten=False, visualize=False)
                        if failure:
                            continue
                        pcd.transform(trans)

                    pcd_collections.append(pcd.points)

            self.initial_translation = stored_trans

        # merge pointclouds
        if pcd_collections:
            combined_array = np.vstack(pcd_collections)
        else:
            # empty array for no points
            combined_array = np.empty((0, 3))
        final_pcd = get_pcd_from_numpy(combined_array)
        return final_pcd


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


    """Functions (adjusted from ChatGPT) for using TSDF integration for Pointcloud extraction"""

    def integrate_tsdf(self, render_data_list, width, height, near, far,
                                     fov, sdf_trunc=0.04):
        """
        Integrate multiple RGB-D frames from PyBullet into a TSDF volume and extract a unified point cloud.

        Parameters:
          - render_data_list: list of dicts, each with keys:
              'color':       np.array (H, W, 3), dtype=np.uint8
              'depth':       np.array (H, W), depth values in meters
              'view_matrix': list of 16 floats (PyBullet view matrix)
          - width, height: image dimensions
          - near, far: near and far clipping distances
          - fov: vertical field-of-view (in degrees) of the camera (constant across frames)
          - voxel_length, sdf_trunc: TSDF parameters

        Returns:
          An Open3D point cloud from the integrated TSDF volume.
        """
        # Precompute intrinsic parameters once.
        fx, fy, cx, cy = self.get_intrinsics(fov, width, height)
        intrinsic = o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy)

        volume = o3d.pipelines.integration.ScalableTSDFVolume(
            voxel_length=self.voxel_size,
            sdf_trunc=sdf_trunc,
            color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8
        )

        for idx, data in enumerate(render_data_list):
            color_np = np.ascontiguousarray(data['color'][:, :, :3])
            depth_ndc = data['depth']
            depth_linear = np.ascontiguousarray(far * near / (far - (far - near) * depth_ndc))
            view_matrix = data['view_matrix']
            # Compute extrinsics from the view matrix.
            extrinsics = self.get_extrinsics(view_matrix)

            # Convert numpy arrays to Open3D images.
            color_o3d = o3d.geometry.Image(color_np)
            depth_o3d = o3d.geometry.Image(depth_linear.astype(np.float32))

            rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                color_o3d, depth_o3d,
                depth_scale=1.0,  # depth in meters
                depth_trunc=far,
                convert_rgb_to_intensity=False
            )
            print(f"Integrating frame {idx}")
            volume.integrate(rgbd, intrinsic, extrinsics)

        pcd = volume.extract_point_cloud()
        return pcd

    @staticmethod
    def get_intrinsics(fov, width, height):
        """
        Compute camera intrinsics given the vertical FOV (in degrees), aspect ratio,
        and image dimensions. According to standard pinhole camera model:

          f_y = (height/2) / tan(fov/2)
          f_x = f_y * aspect   (since aspect = width/height)
          c_x = width/2, c_y = height/2

        References:
          - Songho’s OpenGL Projection Matrix page:
            http://www.songho.ca/opengl/gl_projectionmatrix.html
        """
        # Convert fov to radians
        fov_rad = np.radians(fov)
        fy = (height / 2) / np.tan(fov_rad / 2)
        fx = fy * (width / height)  # := aspect
        cx = width / 2.0
        cy = height / 2.0
        return fx, fy, cx, cy

    @staticmethod
    def get_extrinsics(view_matrix):
        """
        Compute the extrinsic matrix from a PyBullet view matrix.

        The view matrix is given in OpenGL convention (as a list of 16 floats in column-major order).

        Steps:
          1. Reshape the view matrix into a 4x4 matrix (V).
          2. Invert V to get the camera pose (in world coordinates) in OpenGL coordinates.
          3. Apply a correction to account for OpenGL’s coordinate convention:
             multiply by a correction matrix C = diag([1, -1, -1, 1]).
          4. Finally, invert the corrected camera pose to obtain the extrinsic matrix mapping world coordinates to camera coordinates.

        This extrinsic matrix is appropriate to use with Open3D TSDFVolume.integrate().

        References:
          - Discussions on PyBullet forums and OpenGL conventions.
        """
        V = np.array(view_matrix).reshape((4, 4), order='F')
        cam_pose = np.linalg.inv(V)
        # Correction: flip Y and Z axes
        correction = np.diag([1, -1, -1, 1])
        corrected_pose = cam_pose @ correction
        # Now, extrinsics map world -> camera:
        extrinsics = np.linalg.inv(corrected_pose)
        return extrinsics
