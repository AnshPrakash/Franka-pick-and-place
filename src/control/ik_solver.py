import numpy as np
from src.simulation import Simulation
import robotic as ry
import pybullet as p
from scipy.spatial.transform import Rotation as R

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
        table.setPosition(table_pos)
        table.setQuaternion(table_ori)

        # Adjust robot position
        # Has to done after setting the table position because table is the parent of robot
            
        l_panda_base = self.C.getFrame("l_panda_base")
        l_panda_base.setPosition(robot_pos)
        l_panda_base.setQuaternion(robot_ori)

        joint_states = self.sim.robot.get_joint_positions()
        self.C.setJointState(joint_states)


        # for this part keep orientation in pybullet format for calulating relative rotation 
        # to get the orientation in ry_config ee
        ee_pos, ee_ori = self.sim.robot.get_ee_pose() 
        l_gripper = self.C.getFrame("l_gripper")
        grip_ori = l_gripper.getPose()[3:]
        grip_ori = np.array([grip_ori[1], grip_ori[2], grip_ori[3], grip_ori[0]])
        r_grip_ori = R.from_quat(grip_ori)
        r_ee_ori = R.from_quat(ee_ori)
        # Save this relative rotaion
        self.r_rel = (r_grip_ori*r_ee_ori.inv())


        # #DEBUG

        # l_panda_base.setShape(ry.ST.marker, [0.9])
        # b_frame = self.C.addFrame('pybullet-base')
        # b_frame.setShape(ry.ST.marker, [0.6])  # Adjust size as needed
        # b_frame.setPosition(robot_pos)
        # b_frame.setQuaternion(robot_ori)  # [w, x, y, z] convention


        # print(self.C.getFrameNames())
        # l_gripper = self.C.getFrame("l_gripper")
        # l_gripper.setShape(ry.ST.marker, [0.4])  # Adjust size as needed


        # frame = self.C.addFrame('pybullet-ee')
        # ee_pos, ee_ori = self.sim.robot.get_ee_pose()
        # frame.setShape(ry.ST.marker, [0.7])  # Adjust size as needed
        # frame.setPosition(ee_pos)
        # ee_ori = self.get_ry_ori(ee_ori)
        # # ee_ori = [ee_ori[3], ee_ori[0], ee_ori[1], ee_ori[2] ]
        # frame.setQuaternion(ee_ori)  # [w, x, y, z] convention

        # print(l_gripper.getPose())
        # self.C.view(True)
        # exit()

        # Add wall
        wall = self.C.addFrame("wall")
        wall.setShape(ry.ST.ssBox, [10, 10, 0.1, 0.01])
        wall.setPosition(wall_pos)
        wall.setQuaternion(wall_ori)
        wall.setColor([0.5, 0.5, 0.5])
    
    def get_ry_ori(self, orientation):
        """
            args:
                orientation: from pybullet of ee
            Output:
                quat: For ry config of l_gripper
        """
        
        # Create rotation object from the PyBullet quaternion
        r_pb = R.from_quat(orientation)

        r = self.r_rel * r_pb #.as_matrix()

        q = r.as_quat()
        return [q[3], q[0], q[1], q[2]]




    def compute_target_configuration(self, target_pos, target_ori):
        """
           Compute the robot's joint configuration given the target end-effector
           position and orientation
        """
        # Convert target position and orientation to ry coordinate system
        # target_ori = [target_ori[3], target_ori[0], target_ori[1], target_ori[2]]
        target_ori = self.get_ry_ori(target_ori)
        
        # Create a new frame for the debugging target
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
        
        
        # # Get joint limits
        # j_lower, j_upper = self.sim.robot.get_joint_limits()

        # Get Wall and base positions
        wall_pos, wall_orn = p.getBasePositionAndOrientation(self.sim.wall)
        wall_orn = [wall_orn[3], wall_orn[0], wall_orn[1], wall_orn[2]] # Convert to RAI quaternion

        qHome = self.C.getJointState()

        # Initialize KOMO solver

        # The KOMO constructor has arguments:

        # config: the configuration, which is copied once (for IK) or many times (for waypoints/paths) to be the optimization variable
        # phases: the number of phases (which essentially defines the real-valued interval over which objectives can be formulated)
        # slicesPerPhase: the discretizations per phase -> in total we have 
        # configurations which form the path and over which we optimize
        # kOrder: the “Markov-order”, i.e., maximal tuple of configurations over which we formulate features (e.g. take finite differences)
        # enableCollisions: if True, KOMO runs a broadphase collision check (using libFCL) in each optimization step – only then accumulative collision/penetration features will correctly evaluate to non-zero. But this is costly.
        
        komo = ry.KOMO(self.C, 1,1,0, False)
        # Add cosntraints
        #keep the robot near home position
        komo.addObjective([], ry.FS.jointState, [], ry.OT.sos, [1e1], qHome)

        # Minimize collisions
        # komo.addObjective([], ry.FS.accumulatedCollisions, [], ry.OT.eq)
        komo.addObjective([], ry.FS.accumulatedCollisions, [], ry.OT.sos, [1e1])

        # keep the joint limits safe
        komo.addObjective([], ry.FS.jointLimits, [], ry.OT.ineq)

        # Move to the target position
        # the left gripper (`l_gripper`) to `target_pos`
        komo.addObjective([], ry.FS.position, ['l_gripper'], ry.OT.eq, [1e1], target_pos)

        # Set `l_gripper`'s orientation to `target_ori`
        komo.addObjective([], ry.FS.quaternion, ['l_gripper'], ry.OT.eq, [1e1], target_ori)
        # komo.addObjective([], ry.FS.quaternion, ['l_gripper'], ry.OT.sos, [1e1], target_ori)

        # Keep the end-effector above the table
        komo.addObjective([], ry.FS.distance, ['l_gripper', 'l_panda_base'], ry.OT.ineq, [1e1], [0.05])


        # keep the end-effector away from the wall
        
        # komo.addObjective([], ry.FS.positionDiff, ['l_gripper'], ry.OT.ineq, np.diag([-1e1, 0, 0]), [wall_pos[0], 0 ,0 ])
        komo.addObjective([], ry.FS.position, ['l_gripper'], ry.OT.ineq, np.diag([-1e1, 0.0, 0.0]), [wall_pos[0]])
        # komo.addObjective([], ry.FS.position, ['l_gripper'], ry.OT.sos, [1e1], [-0.6, None, None])
        # komo.addObjective(
        #                     [],                  # time slice: apply everywhere or specify particular slices
        #                     ry.FS.positionDiff,  # feature: position difference
        #                     ['l_gripper', 'wall'],   # frames: compare 'point' to 'wall'
        #                     ry.OT.ineq,          # inequality type constraint
        #                     [[-1, 0, 0]],         # scale: project the difference on the x-axis only
        #                     [0]                  # target: 0, which makes (x_point + 6) >= 0
        #                 )

        # Solve for new joint positions & Target position
        ret = ry.NLP_Solver(komo.nlp(), verbose=0 ) .solve()
        if ret.feasible:
            print('-- Solution is feasible!')
        else:
            print('-- THIS IS INFEASIBLE!')
            # return None
        
        q = komo.getPath()

        self.C.setJointState(q[0])
        self.C.view(True)
        return q[0]

