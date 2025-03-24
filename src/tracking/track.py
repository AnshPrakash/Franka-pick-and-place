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
        
        self.aabb_min, self.aabb_max = p.getAABB(obj_id)
        self.object_size = (self.aabb_max[0] - self.aabb_min[0]) / 2 # approx radius of the sphere
        print("Object Size", self.object_size)
        self.perception = Perception()
        
        # Get initial estimate of the object position
        
        measurement = self.measure()
        x,y,z = 0,0,0
        if measurement is not None:
            x, y, z = measurement[0], measurement[1], measurement[2]
        
        self.prev_position = measurement
        # Initialize Kalman Filter for the object
        self.kf_obstacle = self.obstacle_kf( x = x, y = y, z = z,
                                             dt = sim.timestep,
                                             r = 0.5,
                                             q = 0.3)
    
    def measure(self):
        pcb_obj = self.perception.get_pcds( [self.obj_id],
                                             self.sim, 
                                             use_ee = False)
        points = np.asarray(pcb_obj[self.obj_id].points)
        
        # solve least squares to estimate the center of the circle

        
        # Extract x, y, z coordinates
        x = points[:, 0]
        y = points[:, 1]
        z = points[:, 2]

        # Build matrix A and vector b for the system: A * [x_c, y_c, z_c, D]^T = b
        A = np.column_stack((-2*x, -2*y, -2*z, np.ones_like(x)))
        b = -(x**2 + y**2 + z**2)
        # Solve the least squares system
        # The solution vector p contains [x_c, y_c, z_c, D]
        try:
            p, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
            if rank < 4:
                # The system is underdetermined: cannot reliably solve for sphere parameters.
                return None
        except Exception as e:
            print( "Exception in Tracking, can estimate from pcd", e)
            return None
        center = p[:3]
        D = p[3]

        # Compute the radius
        radius = np.sqrt(np.sum(center**2) - D)
        print("Estimated center:", center)
        print("Estimated radius:", radius)  

        return center

    
        
    def estimate_position(self, measurement: list):
        """Get the position of the object in the scene."""
        # Use the segmentation mask, depth image, and RGB image to estimate the position of the object
        # segmentation mask to extract the object
        # Single view of the object

        measurement = self.measure()
        if (measurement is None):
            measurement = self.prev_position
        
        
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
        self.prev_position = estimated_postion
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
