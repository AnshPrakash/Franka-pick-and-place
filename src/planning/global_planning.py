import numpy as np
import robotic as ry
import pybullet as p

from src.simulation import Simulation
from src.control import IKSolver
from src.tracking import Track


class Global_planner(IKSolver):
    def __init__(self, sim: Simulation):
        """
            Initialize KOMO planner with robot configuration.
            Estimate the robot's joint configuration given the target end-effector position 
            and orientation.
        """
        super.__init__(sim)
        
        self.obstacles_tracker = []
        for obstacle in self.sim.obstacles:
            self.obstacles_tracker.append(Track(obstacle.id,sim))

    def get_obstacles(self) -> list:
        """
            Get the center of obstacle centers and their radii
            Output: List[(position, size)]
        """
        obstacles = []
        for tracker in self.obstacles_tracker:
            obstacles.append((tracker.estimate_position(), tracker.object_size))
        return obstacles

        
    def plan(self, target_pos, target_ori):
        """
           Compute the path to goal position
           using RRT*
        """
        target_ori = self.get_ry_ee_ori(target_ori)
        
        # Create a new frame for the debugging target
        
        target_frame = self.C.getFrame("target_marker")
        if target_frame is None:
            target_frame = self.C.addFrame('target_marker')
        target_frame.setShape(ry.ST.marker, [.4])  # Marker is visual only
        target_frame.setPosition(target_pos)
        target_frame.setQuaternion(target_ori)
        target_frame.setColor([0.0, 1.0, 0.0])  # Green marker
        
        # Get current robot state
        joint_states = self.sim.robot.get_joint_positions()

        # Update KOMO with new state
        # Will also update the robot's base and keep the real robot state
        self.C.setJointState(joint_states)

        