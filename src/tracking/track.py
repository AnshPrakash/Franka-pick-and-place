import numpy as np
import pybullet as p
from filterpy.kalman import KalmanFilter

import math
import matplotlib.pyplot as plt

from src.simulation import Simulation
from src.perception import Perception

class Track:
    """Track class for tracking objects given its object id in the scene."""
    def __init__(self,obj_id: int ,sim: Simulation):
        """
            Initialize the Track object.
            
            Args:
                obj_id: Object ID of the object to track.
                sim: Simulation object.
        """
        self.obj_id = obj_id
        self.sim = sim
        self.object_color = None
        self.position = None
        self.object_size = None # Radius of the sphere
        self.perception = Perception()
        self.sensor = Measurement(obj_id, sim)
        # Get initial estimate of the object position
        
        measurement = self.perception.get_pcds( [self.obj_id],
                                                self.sim, 
                                                use_ee = False)
        
        # measurement = self.sensor.measure()
        real_obstacle_world_pos, _ =  p.getBasePositionAndOrientation(self.obj_id)
        print("MEASURE\n")
        points = np.asarray(measurement[self.obj_id].points)
        print(points.shape)
        print("Mean of point cloud", np.median(points, axis=0))
        print("Actual object pos", real_obstacle_world_pos  )
        exit()

        measurement = self.sensor.measure()
        x,y,z = 0,0,0
        if measurement is not None:
            x, y, z = measurement[0], measurement[1], measurement[2]
        
        # Initialize Kalman Filter for the object
        self.kf_obstacle = self.obstacle_kf( x = x, y = y, z = z,
                                             dt = sim.timestep,
                                             r = 0.5,
                                             q = 0.3)
        
    
        
    def estimate_position(self, measurement: list):
        """Get the position of the object in the scene."""
        # Use the segmentation mask, depth image, and RGB image to estimate the position of the object
        # segmentation mask to extract the object
        # Single view of the object

        measurement = self.sensor.measure()        
        return measurement

        x, y, z = measurement[0], measurement[1], measurement[2]

        kf_obstacle = self.kf_obstacle

        # In your simulation loop, suppose you obtain a rough measurement (x_meas, y_meas, z_meas):
        x_meas, y_meas, z_meas = measurement

        z_measure = np.array([[x_meas], [y_meas], [z_meas]])

        # First, predict the next state
        kf_obstacle.predict()

        # Then update the filter with the new measurement
        kf_obstacle.update(z_measure)

        # The updated state can now be used as your best estimate
        estimated_state = kf_obstacle.x
        estimated_postion = estimated_state[[0, 2, 4]].flatten()
        print("Estimated Position with Kalman Filter:", estimated_postion)
        return estimated_postion

    
    def obstacle_kf(self, x: float, y: float, z: float, dt: float, r: float, q: float):
        """
            Initialize a Kalman filter for a 3D obstacle.
            
            Args:
                x, y, z: Initial positions.
                dt: Time step.
                r: Measurement noise scaling factor.
                q: Process noise scaling factor.
                
            Returns:
                Initialized KalmanFilter object.
        """
        kf = KalmanFilter(dim_x=6, dim_z=3)
    
        # State Transition Matrix (F)
        kf.F = np.array([[1, dt, 0,  0, 0,  0],
                        [0,  1, 0,  0, 0,  0],
                        [0,  0, 1, dt, 0,  0],
                        [0,  0, 0,  1, 0,  0],
                        [0,  0, 0,  0, 1, dt],
                        [0,  0, 0,  0, 0,  1]])
        
        # Measurement function H: extracts the positions from state
        kf.H = np.array([[1, 0, 0, 0, 0, 0],
                        [0, 0, 1, 0, 0, 0],
                        [0, 0, 0, 0, 1, 0]])
        
        # Measurement noise covariance R (assumes independent errors in x,y,z)
        kf.R = r * np.eye(3)
        
        # Process noise covariance Q (model uncertainty)
        # kf.Q = q * np.eye(6)
        Q = np.diag([q, 10*q, q, 10*q, q, 10*q])
        kf.Q = Q

        
        # Initialize state vector: start with the measured position and assume zero initial velocity.
        kf.x = np.array([[x], [0],
                        [y], [0],
                        [z], [0]])
        
        return kf


class Measurement:
    def __init__(self, obj_id: int, sim: Simulation):
        """
            Initialize the Measurement object.
            
            Args:
                obj_id: Object ID of the object to track
                sim: Simulation object.
        """
        
        self.sim = sim
        self.fov = 70.0 # from camera settings in config
        self.obj_id = obj_id

    def measure(self):
        """Get the position estimate from the object in the scene."""
        # first static
        rgb, depth, seg = self.sim.get_static_renders()

        # self.visualize_segmentation( rgb, seg,self.obj_id )

        depth_real = self.depth_to_real(depth)
        # Compute camera intrinsics from camera settings
        K = self.compute_intrinsics(self.sim.width,
                                    self.sim.height,
                                    self.fov)
        
        # Extract 3D points corresponding to the target object using the segmentation mask
        object_points = []
        height_img, width_img = depth_real.shape
        mean_u ,mean_v = 0.0 ,0.0
        total = 0
        for v in range(height_img):
            for u in range(width_img):
                if seg[v, u] == self.obj_id:
                    total +=1
                    mean_u += u
                    mean_v += v 
                    point = self.pixel_to_camera_coords(u, v, depth_real, K)
                    object_points.append(point)

        mean_u = mean_u/total
        mean_v = mean_v/total

        object_points = np.array(object_points)
        if len(object_points) == 0:
            print("No object points found for object id", self.obj_id)
            return None

        # Estimate the object's position by computing the centroid of its 3D points
        object_center = np.mean(object_points, axis=0)

        # for Debuging
        print("\n\n======== DEBUG==========")
        view_mat = np.array(self.sim.stat_viewMat).reshape(4, 4)
        proj_mat = np.array(self.sim.projection_matrix).reshape(4, 4).T

        real_obstacle_world_pos, _ =  p.getBasePositionAndOrientation(self.obj_id)
        real_obstacle_pos = np.append(real_obstacle_world_pos,1)
        real_camera_cord = view_mat.dot(real_obstacle_pos)
        

        (u,v), image_cord = self.camera_to_image_coords(
                                               real_camera_cord[:3],
                                               proj_mat,
                                               self.sim.width,
                                               self.sim.height )
        K = self.compute_intrinsics_from_projection()
        print("Error ", (u, v))
        check_obstacle_camera_pos = self.pixel_to_camera_coords(u, v, depth_real, K)

        check_obstacle_word_pos = self.camera_to_world( real_camera_cord[:3], self.sim.stat_viewMat)

        
        # print("NDC coordinates:", image_cord[:3])
        # print("\n\n")
        print("Intrinsic from fov", self.compute_intrinsics(self.sim.width,
                                    self.sim.height,
                                    self.fov), "\n")
        print("Intrinsic from proj", self.compute_intrinsics_from_projection(),"\n")

        print("Pixel coordinates (u, v):", (u, v), "\n")

        # print("Avg. Segmented (mean_u, mean_v)", (mean_u, mean_v))
        print("Actual Camera co-ordinate", real_camera_cord)
        print("Re-map to camera co-ordinate system", check_obstacle_camera_pos , "\n")
        

        # this is correct
        print("World  pos from p", real_obstacle_world_pos)
        print("Map to Camera pos", check_obstacle_word_pos)
        exit()
        object_center = self.camera_to_world( object_center, self.sim.stat_viewMat)
        # print("Estimated 3D Object Position in Camera Coordinates:", object_center)


        return object_center

    def camera_to_image_coords(self, camera_point, projection_matrix, width, height):
        """
        Converts a 3D point from the camera coordinate frame to pixel coordinates in the image.

        Parameters:
        - camera_point: np.array (3,) or (4,) -> The 3D point in camera coordinates.
        - projection_matrix: np.array (4,4)  -> The 4x4 projection matrix.
        - width: int  -> Image width in pixels.
        - height: int -> Image height in pixels.

        Returns:
        - (u, v): tuple (int, int) -> The pixel coordinates in the image.
        - ndc_coords: np.array (3,) -> The normalized device coordinates (NDC).
        """

        # Convert to homogeneous coordinates if needed (ensure it's a 4D vector)
        if len(camera_point) == 3:
            camera_point = np.append(camera_point, 1)

        # Transform to clip space
        clip_coords = projection_matrix @ camera_point  # P * X_camera

        # Perform homogeneous division to get NDC coordinates
        ndc_coords = clip_coords[:3] / clip_coords[-1]  # (x_ndc, y_ndc, z_ndc)

        # Convert NDC to pixel coordinates
        u = int((width / 2) * (ndc_coords[0] + 1))   # Scale & shift x
        v = int((height / 2) * (1 - ndc_coords[1]))  # Scale & invert y

        return (u, v), ndc_coords

    def depth_to_real(self, depth_map, near = 0.01, far = 5.0):
        """
        Convert PyBullet normalized depth map to real-world depth (meters).
        https://stackoverflow.com/questions/6652253/getting-the-true-z-value-from-the-depth-buffer

        Args:
            depth_map (numpy array): Normalized depth map with values in range [0, 1].
            near (float): Near clipping plane distance. Default value from config is 0.01.
            far (float): Far clipping plane distance. Default value from config is 5.0.

        Returns:
            numpy array: Depth map with real-world distances in meters.
        """
        # Conversion formula:
        # depth_real = 2 * far * near / (far + near - depth_norm*(far - near))
        # depth = 2*depth_map - 1
        depth = depth_map
        # depth_real = 2*(far * near) / (far + near - depth * (far - near))
        depth_real = (far * near) / (far  - depth * (far - near))
        return depth_real
    
    def compute_intrinsics_from_projection(self):
        """
        Compute the camera intrinsic matrix (K) from the projection matrix.

        Returns:
            np.array: 3x3 intrinsic camera matrix.
        """
        proj_matrix = np.array(self.sim.projection_matrix).reshape(4, 4) # Transpose for correct shape

        width = self.sim.width
        height = self.sim.height

        fx = proj_matrix[0, 0] * width / 2.0  # Focal length in x
        fy = proj_matrix[1, 1] * height / 2.0  # Focal length in y
        cx = (1 - proj_matrix[0, 2]) * width / 2.0  # Principal point x
        cy = (1 + proj_matrix[1, 2]) * height / 2.0  # Principal point y
        print(proj_matrix)
        K = np.array([
            [fx,  0,  cx],
            [0,  fy,  cy],
            [0,   0,   1]
        ])
        return K
    
    def compute_intrinsics(self, width, height, fov):
        """
        Compute the camera intrinsic matrix given the image size and FOV.
        Assumes FOV is the vertical field-of-view.
        
        Args:
            width (int): Image width in pixels.
            height (int): Image height in pixels.
            fov (float): Vertical field-of-view in degrees.

        Returns:
            numpy array: 3x3 intrinsic matrix.
        """
        fov_rad = math.radians(fov)
        # The focal length (in pixels) for the vertical axis.
        fy = height / (2 * math.tan(fov_rad / 2))
        # Compute fx based on the aspect ratio.
        fx = fy * (width / height)
        cx = width / 2.0
        cy = height / 2.0
        K = np.array([[fx, 0, cx],
                    [0, fy, cy],
                    [0,  0,  1]])
        return K
    
    def pixel_to_camera_coords(self, u, v, depth_map, K):
        """
        Convert an image pixel (u, v) to 3D camera coordinates using the depth map.
        
        Args:
            u (int): Pixel column index.
            v (int): Pixel row index.
            depth_map (numpy array): Depth map with real-world distances.
            K (numpy array): 3x3 camera intrinsic matrix.

        Returns:
            numpy array: 3D point [X, Y, Z] in camera coordinates.
        """
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]
        # u, v = int(round(u)), int(round(v))
        Z = depth_map[u, v]  # real-world depth at pixel (u,v)
        X = (u - cx) * Z / fx
        Y = (v - cy) * Z / fy
        return np.array([X, Y, Z])
    
    def camera_to_world(self, P_camera, view_matrix):
        """
        Transform a point from camera coordinates to world coordinates.
        
        Args:
            P_camera (numpy array): 3D point [X, Y, Z] in camera coordinates.
            view_matrix (list or array): The view matrix as provided by PyBullet.
            
        Returns:
            numpy array: 3D point in world coordinates.
        """
        # Reshape view_matrix into a 4x4 matrix.
        view_mat = np.array(view_matrix).reshape(4, 4)
        # Invert the view matrix to get the extrinsic transformation (camera-to-world)
        extrinsics = np.linalg.inv(view_mat)
        # Convert the camera point to homogeneous coordinates by appending 1.
        P_homog = np.append(P_camera, 1)
        P_world_homog = extrinsics.dot(P_homog)
        return P_world_homog[:3]

    
    def visualize_segmentation(self, rgb, seg, target_obj_id):
        """
        Visualizes the segmentation of an object in the scene.

        Args:
            rgb (numpy array): The RGB image from PyBullet.
            seg (numpy array): The segmentation mask.
            target_obj_id (int): The object ID to highlight.

        Returns:
            None (Displays the image).
        """
        # Create a mask where the object is detected
        object_mask = (seg == target_obj_id)

        # Create a copy of the RGB image
        segmented_rgb = rgb.copy()

        # Apply a red overlay where the object is detected
        segmented_rgb[object_mask] = [255, 0, 0, 255]  # Red color overlay

        # Plot the original and segmented images
        fig, ax = plt.subplots(1, 2, figsize=(12, 6))

        ax[0].imshow(rgb)
        ax[0].set_title("Original RGB Image")
        ax[0].axis("off")

        ax[1].imshow(segmented_rgb)
        ax[1].set_title("Segmented Object (ID={})".format(target_obj_id))
        ax[1].axis("off")

        plt.show()
