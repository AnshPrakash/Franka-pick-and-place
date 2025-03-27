import numpy as np
from src.simulation import Simulation
from src.control import IKSolver
import robotic as ry
import pybullet as p


class Global_planner(IKSolver):
    def __init__(self, sim: Simulation):
        """
            Initialize KOMO planner with robot configuration.
            Estimate the robot's joint configuration given the target end-effector position 
            and orientation.
        """
        super.__init__(sim)
        self.obstacles = {"ob1" : {}, "ob2" : {} }

    def update_obstacles(self):
        
        pass
        
        
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

        