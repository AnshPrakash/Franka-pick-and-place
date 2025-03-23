from typing import Dict, Any, Optional

import numpy as np
import pybullet as p
import pybullet_data


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
    
    def estimate_depth(self):
        """Get the depth of the object in the scene."""
        
        pass

    def estimate_position(self):
        """Get the position of the object in the scene."""
        # Use the segmentation mask, depth image, and RGB image to estimate the position of the object
        # segmentation mask to extract the object
        # Single view of the object

        # self.position = self.sim.get_object_position(self.obj_id)
        return self.position

