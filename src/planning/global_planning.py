import numpy as np
from src.simulation import Simulation
import robotic as ry
import pybullet as p


class Global_planner:
    def __init__(self, sim: Simulation):
        """
            Initialize KOMO planner with robot configuration.
            Estimate the robot's joint configuration given the target end-effector position 
            and orientation.
        """
        self.sim = sim

        C = ry.Config()
        C.addFile(ry.raiPath('scenarios/pandaSingle.g'))

        self.C = C
        self._configure_ry()
        

    def _configure_ry(self):
        """
            Align the ry configuration with the pybullet simulation
        """
        
        table_pos, table_ori = p.getBasePositionAndOrientation(self.sim.table.id)

        tray_pos, tray_ori = p.getBasePositionAndOrientation(self.sim.goal.id)

        robot_pos, robot_ori = self.sim.robot.pos, self.sim.robot.ori

        wall_pos, wall_ori = p.getBasePositionAndOrientation(self.sim.wall)
        

        # Convert PyBullet quaternion [x, y, z, w] → RAI quaternion [w, x, y, z]
        table_ori = [table_ori[3], table_ori[0], table_ori[1], table_ori[2]]
        tray_ori = [tray_ori[3], tray_ori[0], tray_ori[1], tray_ori[2]]
        robot_ori = [robot_ori[3], robot_ori[0], robot_ori[1], robot_ori[2]]  
        wall_ori = [wall_ori[3], wall_ori[0], wall_ori[1], wall_ori[2]]

        # Create the tray frame
        # Get collision shape data (gives more accurate size)
        tray_id = self.sim.goal.id
        collision_data = p.getCollisionShapeData(tray_id, -1)
        tray_size = collision_data[0][3]  # Extracting halfExtents (for box shapes)
        tray = self.C.addFrame("tray")
        tray.setShape(ry.ST.ssBox, [tray_size[0], tray_size[1], tray_size[2], 0.02])  # Full size, with corner radius
        tray.setColor([0.5, 0.5, 0.5])  # Grey color
        tray.setPosition(tray_pos)  
        tray.setQuaternion(tray_ori)

        # Create the table frame
        table = self.C.getFrame("table")
        table.setShape(ry.ST.ssBox, [2.5, 2.5, 0.05, 0.01])  # Full size, with corner radius
        table.setColor([0.2])  # Grey color


        table_pos = [table_pos[0], table_pos[1], self.sim.robot.tscale * 0.6] # 0.6 table height came from urdf file
        self.C.getFrame("table").setPosition(table_pos)
        self.C.getFrame("table").setQuaternion(table_ori)

        # Adjust robot position
        # Has to done after setting the table position because table is the parent of robot
            
        l_panda_base = self.C.getFrame("l_panda_base")
        l_panda_base.setPosition(robot_pos)
        l_panda_base.setQuaternion(robot_ori)

        joint_states = self.sim.robot.get_joint_positions()
        self.C.setJointState(joint_states)

        # Add wall
        wall = self.C.addFrame("wall")
        wall.setShape(ry.ST.ssBox, [10, 10, 0.1, 0.01])
        wall.setPosition(wall_pos)
        wall.setQuaternion(wall_ori)
        wall.setColor([0.5, 0.5, 0.5])
