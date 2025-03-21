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
            return None

        q = komo.getPath()
        
        return q[0]

