from typing import Dict, Any, Optional

import numpy as np
import pybullet as p
from filterpy.kalman import KalmanFilter


from src.robot import Robot
from src.objects import Obstacle, Table, Box, YCBObject, Goal
from src.utils import pb_image_to_numpy
from src.simulation import Simulation

class Track:
    """Track class for tracking objects given its object id in the scene."""
    def __init__(self,obj_id: int ,sim: Simulation):
        self.obj_id = obj_id
        self.sim = sim
        self.object_color = None
        self.position = None
        self.object_size = None # Radius of the sphere
        
        self.kf_obstacle = self.obstacle_kf(x=0, y=0, z=0, dt=sim.timestep, r=0.5, q=0.3)
    
    def measurement(self):
        """Get the depth of the object in the scene."""
        
        pass

    def estimate_position(self, measurement: list):
        """Get the position of the object in the scene."""
        # Use the segmentation mask, depth image, and RGB image to estimate the position of the object
        # segmentation mask to extract the object
        # Single view of the object


        x, y, z = measurement
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
        print("Estimated Position:", estimated_state[[0, 2, 4]].flatten())

    
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
