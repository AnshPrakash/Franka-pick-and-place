import numpy as np
from src.simulation import Simulation
import robotic as ry

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
        Compute the robot's joint configuration given the target end-effector position and orientation.
        """

        # Get current robot state
        joint_states = self.sim.robot.get_joint_positions()

        # Update KOMO with new state
        self.C.setJointState(joint_states)

        # Initialize KOMO solver
        komo = ry.KOMO()
        komo.setConfig(self.C, True)


        # Get joint limits
        j_lower, j_upper = self.sim.robot.get_joint_limits()

        # Get Wall and base positions
        base_pos = self.sim.get_base_position()
        table_pos = self.sim.get_table_position()

        # Get current state
        joint_states = self.sim.robot.get_joint_positions()

    
        # Define optimization problem
        komo.set_end_effector_target(target_pos, target_ori)

        # Solve for new joint positions & Target position
        ret = ry.NLP_Solver(komo.nlp(), verbose=0 ) .solve()
        if ret.feasible:
            print('-- Always check feasibility flag of NLP solver return')
        else:
            print('-- THIS IS INFEASIBLE!')

        q = komo.getPath()

