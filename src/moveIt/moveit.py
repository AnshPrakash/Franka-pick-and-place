import numpy as np

from src.planning import Global_planner
from src.simulation import Simulation


class MoveIt:
    """
        Utilises Global planner iteratively to avoid collision while actually 
        moving the end-effector in the simulation
    """
    def __init__(self, sim: Simulation):
        self.sim = sim
        self.planner = Global_planner(sim)
    

    def moveTo(self, goal_position, goal_ori):
        """
        Args:
            goal_position: Desired end-effector position.
            goal_ori: Desired end-effector orientation in PyBullet quaternion format.
        
        Returns:
            True: if goal pose possible and tries its best to EE without collision
            False: if goal pose is not possible according to control module
        """
        
        replan_freq = 1 # hyperparameter
        MAX_ITER = 200
        epsilon = 0.05
        gp = self.planner

        # Get the target joint configuration
        qT = self.planner.compute_target_configuration(goal_position, goal_ori)
        if qT is None:
            print("IK failed to compute a goal configuration.")
            return False
        qT = np.array(qT)
        last_configuration = self.sim.robot.get_joint_positions()
        check_new_config_freq = 10
        iter = 0
        for i in range(MAX_ITER):
            if i%replan_freq == 0:
                path = gp.plan(goal_position, goal_ori)
                skip_to_config = 10 # hyperparameter >= 1 [0 implies current congifuration]
                if len(path) > skip_to_config:
                    q = path[skip_to_config]
                else:
                    q = path[1]
                self.sim.robot.position_control(q)
            self.sim.step()
            iter += 1
            
            # Check goal pose achieved
            robot_joint_config = self.sim.robot.get_joint_positions()
            # joint_velocities = self.sim.robot.get_joint_velocites()
            # max_abs_diff = np.max(np.abs(robot_joint_config - qT))
            # config_norm = np.linalg.norm(robot_joint_config - qT)
            # velocity_norm = np.linalg.norm(joint_velocities)
            # print("ITER:", iter)
            # print("MAX ABS DIFF: ", max_abs_diff)
            # print("Config NORM: ", config_norm)
            # print("Velocity Norm: ", velocity_norm)
            if i !=0 and i%check_new_config_freq == 0:
                delta_config = np.abs(robot_joint_config - last_configuration)
                last_configuration = robot_joint_config
                print("Change in config",  delta_config )
                if (np.max(delta_config) < epsilon):
                    print(f"Configuration Achieved: No big change in joint configuration in last {check_new_config_freq} steps")
                    break
            # print("================")

        return True