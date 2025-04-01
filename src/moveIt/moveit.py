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
    

    def moveToOld(self, goal_position, goal_ori, joint_space_crit=False):
        """
        Args:
            goal_position: Desired end-effector position.
            goal_ori: Desired end-effector orientation in PyBullet quaternion format.
        
        Returns:
            True: if goal pose possible and tries its best to EE without collision
            False: if goal pose is not possible according to control module
        """

        replan_freq = 5  # hyperparameter
        MAX_ITER = 50
        gp = self.planner

        # Get the target joint configuration
        qT = self.planner.compute_target_configuration(goal_position, goal_ori)
        if qT is None:
            print("IK failed to compute a goal configuration.")
            return False
        qT = np.array(qT)

        last_configuration = self.sim.robot.get_joint_positions()
        last_pos, last_ori = self.sim.robot.get_ee_pose()

        check_new_config_freq = 10
        iter = 0
        for i in range(MAX_ITER):
            if i % replan_freq == 0:
                path = gp.plan(goal_position, goal_ori)
                skip_to_config = 10  # hyperparameter >= 1 [0 implies current congifuration]
                if len(path) > skip_to_config:
                    q = path[skip_to_config]
                else:
                    q = path[1]
                self.sim.robot.position_control(q)
            self.sim.step()
            iter += 1

            if joint_space_crit:
                epsilon = 0.03  # hyperparameter
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
                if i != 0 and i % check_new_config_freq == 0:
                    delta_config = np.abs(robot_joint_config - last_configuration)
                    last_configuration = robot_joint_config
                    print("Change in config", delta_config)
                    if (np.max(delta_config) < epsilon):
                        print(
                            f"Configuration Achieved: No big change in joint configuration in last {check_new_config_freq} steps")
                        break
                # print("================")
            else:
                espilon = 0.03
                curr_pos, curr_ori = self.sim.robot.get_ee_pose()
                if i != 0 and i % check_new_config_freq == 0:
                    if (np.linalg.norm(curr_pos - last_pos) < espilon and
                            np.linalg.norm(curr_ori - last_ori) < espilon):
                        print(i)
                        break
                    last_pos, last_ori = curr_pos, curr_ori

        return True

    def moveTo(self, goal_position, goal_ori, joint_space_crit=False):
        """
        Args:
            goal_position: Desired end-effector position.
            goal_ori: Desired end-effector orientation in PyBullet quaternion format.

        Returns:
            True: if goal pose possible and tries its best to EE without collision
            False: if goal pose is not possible according to control module
        """

        # HYPERPARAMETERS
        replan_freq = 13
        MAX_ITER = 100
        next_q_threshold = 0.5
        epsilon = 0.03  # for position/config change
        check_new_config_freq = 10

        gp = self.planner

        # Get the target joint configuration
        qT = self.planner.compute_target_configuration(goal_position, goal_ori)
        if qT is None:
            print("IK failed to compute a goal configuration.")
            return False
        qT = np.array(qT)

        last_configuration = self.sim.robot.get_joint_positions()
        last_pos, last_ori = self.sim.robot.get_ee_pose()

        iter = 0
        for i in range(MAX_ITER):
            if i != 0 and i % check_new_config_freq == 0:
                if joint_space_crit:
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
                    delta_config = np.linalg.norm(robot_joint_config - last_configuration)
                    delta_goal = np.linalg.norm(qT - robot_joint_config)
                    last_configuration = robot_joint_config
                    print("Change in config", delta_config)
                    if delta_goal < epsilon:
                        print("Goal Achieved")
                        break
                    if delta_config < epsilon:
                        print(
                            f"Configuration Achieved: No big change in joint configuration in last {check_new_config_freq} steps")
                        break
                # print("================")
                else:
                    curr_pos, curr_ori = self.sim.robot.get_ee_pose()
                    if (np.linalg.norm(curr_pos - goal_position) < epsilon and
                            np.linalg.norm(curr_ori - goal_ori) < epsilon):
                        print(f"Goal Achieved")
                        break
                    if (np.linalg.norm(curr_pos - last_pos) < epsilon and
                            np.linalg.norm(curr_ori - last_ori) < epsilon):
                        print(
                            f"Position Achieved: No big change in position in last {check_new_config_freq} steps")
                        break
                    last_pos, last_ori = curr_pos, curr_ori

            if i % replan_freq == 0:
                path = gp.plan(goal_position, goal_ori)
                if path is None:
                    if i == 0:
                        print("No path found")
                        return False
                    # if i not 0 we already moved the robot and hope its in a relatively close position
                    return True
                initial_q = path[0]
                q = path[-1]
                for next_q in path[1:]:
                    if np.linalg.norm(initial_q - next_q) > next_q_threshold:
                        q = next_q
                        break
                self.sim.robot.position_control(q)
            self.sim.step()
            iter += 1

        return True

    def moveToSmooth(self, goal_position, goal_ori, joint_space_crit=False):
        """
        Instead of using a fixed point within the RRT plan for a defined number of steps before replanning, update the
        waypoint to the next point in the RRT plan if the distance is lower than a threshold and we are not replanning yet.
        Args:
            goal_position: Desired end-effector position.
            goal_ori: Desired end-effector orientation in PyBullet quaternion format.

        Returns:
            True: if goal pose possible and tries its best to EE without collision
            False: if goal pose is not possible according to control module
        """

        # HYPERPARAMETERS
        replan_freq = 20
        MAX_ITER = 100
        next_q_threshold = 1.0
        epsilon = 0.03  # for position/config change
        check_new_config_freq = 10

        gp = self.planner

        # Get the target joint configuration
        qT = self.planner.compute_target_configuration(goal_position, goal_ori)
        if qT is None:
            print("IK failed to compute a goal configuration.")
            return False
        qT = np.array(qT)

        last_configuration = self.sim.robot.get_joint_positions()
        last_pos, last_ori = self.sim.robot.get_ee_pose()

        # initialize important variables
        replan_counter = 0 # initially we need to plan
        path = None
        waypoint = None # index for current waypoint q
        q = None
        for i in range(MAX_ITER):
            robot_joint_config = self.sim.robot.get_joint_positions()
            curr_pos, curr_ori = self.sim.robot.get_ee_pose()

            if i != 0 and i % check_new_config_freq == 0:
                if joint_space_crit:
                    # Check goal pose achieved

                    # joint_velocities = self.sim.robot.get_joint_velocites()
                    # max_abs_diff = np.max(np.abs(robot_joint_config - qT))
                    # config_norm = np.linalg.norm(robot_joint_config - qT)
                    # velocity_norm = np.linalg.norm(joint_velocities)
                    # print("ITER:", iter)
                    # print("MAX ABS DIFF: ", max_abs_diff)
                    # print("Config NORM: ", config_norm)
                    # print("Velocity Norm: ", velocity_norm)
                    delta_config = np.linalg.norm(robot_joint_config - last_configuration)
                    delta_goal = np.linalg.norm(qT - robot_joint_config)
                    last_configuration = robot_joint_config
                    print("Change in config", delta_config)
                    if delta_goal < epsilon:
                        print("Goal Achieved")
                        break
                    if delta_config < epsilon:
                        print(
                            f"Configuration Achieved: No big change in joint configuration in last {check_new_config_freq} steps")
                        break
                # print("================")
                else:
                    if (np.linalg.norm(curr_pos - goal_position) < epsilon and
                            np.linalg.norm(curr_ori - goal_ori) < epsilon):
                        print(f"Goal Achieved")
                        break
                    if (np.linalg.norm(curr_pos - last_pos) < epsilon and
                            np.linalg.norm(curr_ori - last_ori) < epsilon):
                        print(
                            f"Position Achieved: No big change in position in last {check_new_config_freq} steps")
                        break
                    last_pos, last_ori = curr_pos, curr_ori

            if replan_counter == 0:
                replan_counter = replan_freq
                path = gp.plan(goal_position, goal_ori)
                if path is None:
                    if i == 0:
                        print("No path found")
                        return False
                    # if i not 0 we already moved the robot and hope its in a relatively close position
                    return True
                initial_q = path[0]
                q = path[-1]
                waypoint = len(path) - 1
                for idx, next_q in enumerate(path[1:]):
                    if np.linalg.norm(initial_q - next_q) > next_q_threshold:
                        q = next_q
                        waypoint = idx + 1
                        break
                self.sim.robot.position_control(q)
            self.sim.step()

            # at the end (to ensure waypoint is initialized) check if path waypoint is already reached
            if np.linalg.norm(q - robot_joint_config) < epsilon:
                # update to next waypoint
                if waypoint < len(path) - 1:
                    initial_q = path[waypoint]
                    q = path[-1]
                    waypoint = len(path) - 1
                    for idx, next_q in enumerate(path[waypoint + 1:]):
                        if np.linalg.norm(initial_q - next_q) > next_q_threshold:
                            q = next_q
                            waypoint = idx + waypoint + 1
                            break
                    self.sim.robot.position_control(q)
                # if last waypoint is reached check again for goal, otherwise replan earlier
                else:
                    if (np.linalg.norm(curr_pos - goal_position) < epsilon and
                            np.linalg.norm(curr_ori - goal_ori) < epsilon):
                        print(f"Goal Achieved")
                        break
                    replan_counter = 1 # 1 because it is subtracted at the end of the loop

            replan_counter -= 1

        return True