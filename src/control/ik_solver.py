import numpy as np
from src.simulation import Simulation
import robotic as ry
import pybullet as p

class IKSolver:
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
        # self.C.view()

        self.C.view(pause=True, message="Move the camera and press a key!")
        

    def _configure_ry(self):
        """
            Align the ry configuration with the pybullet simulation
        """
        
        table_pos, table_ori = p.getBasePositionAndOrientation(self.sim.table.id)

        tray_pos, tray_ori = p.getBasePositionAndOrientation(self.sim.goal.id)

        robot_pos, robot_ori = self.sim.robot.pos, self.sim.robot.ori
        

        # Convert PyBullet quaternion [x, y, z, w] → RAI quaternion [w, x, y, z]
        table_ori = [table_ori[3], table_ori[0], table_ori[1], table_ori[2]]
        tray_ori = [tray_ori[3], tray_ori[0], tray_ori[1], tray_ori[2]]
        robot_ori = [robot_ori[3], robot_ori[0], robot_ori[1], robot_ori[2]]  


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
        table.setShape(ry.ST.ssBox, [1.5, 1.5, 0.05, 0.01])  # Full size, with corner radius
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


        # DEBUG
        print("Expected Robot pos", robot_pos)
        print("Expected tray pos", tray_pos)
        print("Expected table pos", table_pos)

        # Real Pos in ry.Config
        print("Real Robot pos", l_panda_base.getPosition())
        print("Real tray pos", tray.getPosition())
        print("Real table pos", table.getPosition())
        

        

        
        # print(" Config adjusted ",self.C.getJointState())
        
        

    def get_coordinate_for_ry(self, target_pos, target_ori):
        """
            Convert the target position and orientation to the ry coordinate system
        """
        target_pos = np.array(target_pos)
        target_ori = np.array(target_ori)
        return target_pos, target_ori

    def compute_target_configuration(self, target_pos, target_ori):
        """
           Compute the robot's joint configuration given the target end-effector
           position and orientation
        """

        # Get current robot state
        joint_states = self.sim.robot.get_joint_positions()

        
        # Update KOMO with new state
        # Will also update the robot's base and keep the real robot state
        self.C.setJointState(joint_states)
        # DEBUG
        # print("DEBUG | DEBUG")
        # for f in self.C.getFrames():
        #     print(f.name, f.asDict()) #info returns all attributes, similar to what is defined in the .g-files
        #     #see also all the f.get... methods
        
        

        # # Get joint limits
        # j_lower, j_upper = self.sim.robot.get_joint_limits()

        # Get Wall and base positions
        base_pos = self.sim.robot.pos[2]
        wall_pos, wall_orn = p.getBasePositionAndOrientation(self.sim.wall)


        qHome = self.C.getJointState()

        # Initialize KOMO solver

        # C → The robot's configuration (environment).
        # T=1 → Only one time step (static optimization).
        # k=1 → Only considers one configuration (not a full trajectory).
        # order=0 → No velocity/smoothness constraints (static pose).
        # verbose=True → Enables logging/debugging.
        
        # komo = ry.KOMO(self.C, T=1, k=1, order=0, verbose=False)
        komo = ry.KOMO(self.C, 1,1,0, True)
        # Add cosntraints
        #keep the robot near home position
        komo.addObjective([], ry.FS.jointState, [], ry.OT.sos, [1e-1], qHome)

        # Minimize collisions
        komo.addObjective([], ry.FS.accumulatedCollisions, [], ry.OT.eq)

        # keep the joint limits safe
        komo.addObjective([], ry.FS.jointLimits, [], ry.OT.ineq)

        # Move to the target position
        # the left gripper (`l_gripper`) to `target_pos`
        komo.addObjective([], ry.FS.position, ['l_gripper'], ry.OT.eq, [1e1], target_pos)

        # Set `l_gripper`'s orientation to `target_ori`
        komo.addObjective([], ry.FS.quaternion, ['l_gripper'], ry.OT.eq, [1e1], target_ori)

        # Keep the end-effector above the table
        komo.addObjective([], ry.FS.position, ['l_gripper'], ry.OT.ineq, [1e1], [0, 0, base_pos + 0.01])

        # keep the end-effector away from the wall
        komo.addObjective([], ry.FS.position, ['l_gripper'], ry.OT.ineq, [1e1], [wall_pos[0] - 0.01, 0, 0 ])

        # Solve for new joint positions & Target position
        ret = ry.NLP_Solver(komo.nlp(), verbose=0 ) .solve()
        if ret.feasible:
            print('-- Always check feasibility flag of NLP solver return')
        else:
            print('-- THIS IS INFEASIBLE!')
            # return None
        
        q = komo.getPath()
        self.C.setJointState(q[0])
        print("DEBUG | DEBUG")
        for f in self.C.getFrames():
            if f.name == "l_gripper" or f.name == "l_panda_base":
                print(f.name, f.asDict()) 

        # self.C.view()

        return q[0]

