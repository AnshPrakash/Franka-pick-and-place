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
        C.addFile(ry.raiPath('scenarios/panda_irobman.g'))

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
        

        world = self.C.getFrame("world")
        world.setShape(ry.ST.marker, [0.6])
        world.setContact(0)


        # Convert PyBullet quaternion [x, y, z, w] → RAI quaternion [w, x, y, z]
        table_ori = [table_ori[3], table_ori[0], table_ori[1], table_ori[2]]
        tray_ori = [tray_ori[3], tray_ori[0], tray_ori[1], tray_ori[2]]
        robot_ori = [robot_ori[3], robot_ori[0], robot_ori[1], robot_ori[2]]
        wall_ori = [wall_ori[3], wall_ori[0], wall_ori[1], wall_ori[2] ]
        

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
        tray.setShape(ry.ST.marker, [0.6])
        tray.setContact(0)

        # Create the table frame
        table = self.C.getFrame("table")
        table.setShape(ry.ST.ssBox, [2.5, 2.5, 0.05, 0.01])  # Full size, with corner radius
        table.setColor([0.2])  # Grey color
        table.setShape(ry.ST.marker, [1.4])
        table.setContact(0)


        table_pos = [table_pos[0], table_pos[1], self.sim.robot.tscale * 0.6] # 0.6 table height came from urdf file
        table.setPosition(table_pos)
        table.setQuaternion(table_ori)

        # Adjust robot position
        # Has to done after setting the table position because table is the parent of robot
            
        l_panda_base = self.C.getFrame("l_panda_base")
        l_panda_base.setPosition(robot_pos)
        l_panda_base.setQuaternion(robot_ori)
        l_panda_base.setShape(ry.ST.marker, [0.6])

        joint_states = self.sim.robot.get_joint_positions()
        self.C.setJointState(joint_states)

        # Add wall
        wall = self.C.addFrame("wall")
        wall.setShape(ry.ST.ssBox, [10, 10, 0.1, 0.01])
        wall.setPosition(wall_pos)
        wall.setQuaternion(wall_ori)
        wall.setColor([0.5, 0.5, 0.5])
        wall.setContact(1)
    

        # #DEBUG

        # l_panda_base.setShape(ry.ST.marker, [0.9])
        # b_frame = self.C.addFrame('pybullet-base')
        # b_frame.setShape(ry.ST.marker, [0.6])  # Adjust size as needed
        # b_frame.setPosition(robot_pos)
        # b_frame.setQuaternion(robot_ori)  # [w, x, y, z] convention


        # print(self.C.getFrameNames())
        l_gripper = self.C.getFrame("l_gripper")
        l_gripper.setShape(ry.ST.marker, [0.4])  # Adjust size as needed
        # print("Gripper position",  l_gripper.getPosition())
        # print("EE position", self.sim.robot.get_ee_pose()[0] )



        # frame = self.C.addFrame('pybullet-ee')
        # ee_pos, ee_ori = self.sim.robot.get_ee_pose()
        # frame.setShape(ry.ST.marker, [0.7])  # Adjust size as needed
        # frame.setPosition(ee_pos)
        # ee_ori = self.get_ry_ee_ori(ee_ori)
        # # ee_ori = [ee_ori[3], ee_ori[0], ee_ori[1], ee_ori[2] ]
        # frame.setQuaternion(ee_ori)  # [w, x, y, z] convention
        
        # print(l_gripper.getPose())

        
    def get_ry_ee_ori(self, orientation):
        """
        Convert a PyBullet end-effector orientation quaternion to the corresponding 
        RAi configuration orientation for the 'l_gripper' frame.

        Args:
            orientation: A quaternion from PyBullet for the end-effector in [x, y, z, w] order.
        
        Returns:
            A quaternion for the RAi configuration of l_gripper in [w, x, y, z] order.
        """
        # Create a Rotation object from the PyBullet quaternion (format: [x, y, z, w])
        r_pb = R.from_quat(orientation)  
        
        # Define the transformation matrix T:
        # This matrix swaps the x and y axes and flips the z axis.
        T = np.array([[0, 1, 0],
                    [1, 0, 0],
                    [0, 0, -1]])
            
        # Create a rotation object from the transformation matrix
        T_rot = R.from_matrix(T)

        # Apply the stored relative rotation (r_rel was computed as: 
        # r_rel = r_grip_ori * r_ee_ori.inv() to convert from ee orientation to l_gripper orientation)
        # Note: multiple T_rot after r_pb because its relative to the local frame
        r = r_pb * T_rot
        # Get the resulting quaternion in PyBullet convention ([x, y, z, w])
        q = r.as_quat()

        # Convert to RAi convention ([w, x, y, z])
        q_rai = [q[3], q[0], q[1], q[2]]
        
        
        return q_rai

    def debug_object(self, object_id, vertices, triangles):
        # Create a new frame for the object
        obj_frame = self.C.addFrame("YcbObject")

        
        # Set the mesh for the object frame
        obj_frame.setMesh(vertices, triangles)

        # Set the pose of the object frame (example values)
        # obj_position, obj_orientation = p.getBasePositionAndOrientation(object_id)
        # obj_orientation = [obj_orientation[3], obj_orientation[0], obj_orientation[1], obj_orientation[2]]
        # obj_frame.setPosition(obj_position)
        # obj_frame.setQuaternion(obj_orientation)  # RAi convention: [w,x,y,z]

        # Optionally, assign a color or other visual properties
        obj_frame.setColor([0.0, 1.0, 0.0])  # Green color
        self.C.view(True)


    def debug(self, marker_name, target_pos, target_ori):
        target_ori = self.get_ry_ee_ori(target_ori)
        target_frame = self.C.getFrame(marker_name)
        if target_frame is None:
            target_frame = self.C.addFrame(marker_name)
        target_frame.setShape(ry.ST.marker, [.4])  # Marker is visual only
        target_frame.setPosition(target_pos)
        target_frame.setQuaternion(target_ori)
        target_frame.setColor([0.0, 1.0, 0.0])  # Green marker
        self.C.view(True)



    def compute_target_configuration(self, target_pos, target_ori = None, convert_ori_to_ry = True):
        """
           Compute the robot's joint configuration given the target end-effector
           position and orientation
        """
        # Convert target position and orientation to ry coordinate system
        if target_ori is not None and  convert_ori_to_ry:
            target_ori = self.get_ry_ee_ori(target_ori)
        
        # Create a new frame for the debugging target
        
        target_frame = self.C.getFrame("target_marker")
        if target_frame is None:
            target_frame = self.C.addFrame('target_marker')
        target_frame.setShape(ry.ST.marker, [.4])  # Marker is visual only
        target_frame.setPosition(target_pos)
        if target_ori is not None:
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
        wall_pos, _ = p.getBasePositionAndOrientation(self.sim.wall)


        qHome = self.C.getJointState()

        # self.C.view(True)

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
        #komo.addObjective([], ry.FS.quaternion, ['l_gripper'], ry.OT.eq, [1e1], target_ori)
        if target_ori is not None:
            komo.addObjective([], ry.FS.quaternion, ['l_gripper'], ry.OT.sos, [1e3], target_ori)

        # Keep the end-effector above the table
        # Since the box is defined using half extents, its top surface is at:
        komo.addObjective([], ry.FS.position, ['l_gripper'], ry.OT.ineq, np.diag([0.0, 0.0, -1e1]), [0,0, 1.0])
        # komo.addObjective([], ry.FS.distance, ['l_gripper', 'l_panda_base'], ry.OT.ineq, [1e1], [0.05])


        # keep the end-effector away from the wall
        
        # komo.addObjective([], ry.FS.positionDiff, ['l_gripper'], ry.OT.ineq, np.diag([-1e1, 0, 0]), [wall_pos[0], 0 ,0 ])
        komo.addObjective([], ry.FS.position, ['l_gripper'], ry.OT.ineq, np.diag([-1e1, 0.0, 0.0]), [wall_pos[0]])
        # komo.addObjective([], ry.FS.position, ['l_gripper'], ry.OT.sos, [-1e1], [-0.6, None, None])
        # komo.addObjective(
        #                     [],                  # time slice: apply everywhere or specify particular slices
        #                     ry.FS.positionDiff,  # feature: position difference
        #                     ['l_gripper', 'wall'],   # frames: compare 'point' to 'wall'
        #                     ry.OT.ineq,          # inequality type constraint
        #                     [[-1, 0, 0]],         # scale: project the difference on the x-axis only
        #                     [0]                  # target: 0, which makes (x_point + 6) >= 0
        #                 )

        # Solve for new joint positions & Target position
        ret = ry.NLP_Solver(komo.nlp(), verbose=0 ).solve()
        if ret.feasible:
            print('-- Solution is feasible!')
        else:
            print('-- THIS IS INFEASIBLE!')
            return None
        
        q = komo.getPath()

        # # DEBUGGING
        # self.C.setJointState(q[0])
        # self.C.view(True)


        return q[0]

