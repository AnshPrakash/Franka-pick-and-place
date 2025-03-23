from typing import Dict, Any, Optional

import numpy as np
import pybullet as p
from filterpy.kalman import KalmanFilter
import math
import matplotlib.pyplot as plt


from src.robot import Robot
from src.objects import Obstacle, Table, Box, YCBObject, Goal
from src.utils import pb_image_to_numpy
from src.simulation import Simulation

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
        self.sensor = Measurement(obj_id, sim)
        # Get initial estimate of the object position
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
        for v in range(height_img):
            for u in range(width_img):
                if seg[v, u] == self.obj_id:
                    point = self.pixel_to_camera_coords(u, v, depth_real, K)
                    object_points.append(point)

        object_points = np.array(object_points)
        if len(object_points) == 0:
            print("No object points found for object id", self.obj_id)
            return None

        # Estimate the object's position by computing the centroid of its 3D points
        object_center = np.mean(object_points, axis=0)
        print("Estimated 3D Object Position in Camera Coordinates:", object_center)

        return object_center


    def depth_to_real(self, depth_map, near = 0.01, far = 5.0):
        """
        Convert PyBullet normalized depth map to real-world depth (meters).

        Args:
            depth_map (numpy array): Normalized depth map with values in range [0, 1].
            near (float): Near clipping plane distance. Default value from config is 0.01.
            far (float): Far clipping plane distance. Default value from config is 5.0.

        Returns:
            numpy array: Depth map with real-world distances in meters.
        """
        # Conversion formula:
        # depth_real = 2 * far * near / (far + near - depth_norm*(far - near))
        depth_real = (2.0 * far * near) / (far + near - depth_map * (far - near))
        return depth_real
    
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
        u, v = int(round(u)), int(round(v))
        Z = depth_map[v, u]  # real-world depth at pixel (u,v)
        X = (u - cx) * Z / fx
        Y = (v - cy) * Z / fy
        return np.array([X, Y, Z])
    
    
    def visualize_segmentation(self, rgb, seg, target_obj_id):
        import matplotlib.pyplot as plt
        import numpy as np

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

    # Example Usage (assuming you have rgb and seg from get_static_renders)
    # visualize_segmentation(rgb, seg, target_obj_id)
