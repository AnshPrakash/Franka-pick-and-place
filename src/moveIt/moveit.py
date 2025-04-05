import numpy as np

from src.planning import Global_planner
from src.simulation import Simulation
import pybullet as p
from scipy.spatial.transform import Rotation as R

class MoveIt:
    """
        Utilises Global planner iteratively to avoid collision while actually 
        moving the end-effector in the simulation
    """
    def __init__(self, sim: Simulation):
        self.sim = sim
        self.planner = Global_planner(sim)
    

    def goal_sampler(self):
        """
            Sample goal position
            Sample goal orientation
        """
        tray_pos, tray_ori = p.getBasePositionAndOrientation(self.sim.goal.id)
        tray_id = self.sim.goal.id
        collision_data = p.getCollisionShapeData(tray_id, -1)
        tray_size = collision_data[0][3]  # Extracting halfExtents (for box shapes)
        tray_size = np.array(tray_size)
        hx, hy, hz = tray_size  # half extents along x, y, and z.

        n_points = 100

        # Uniformly sample x in [-hx, hx] and y in [-hy, hy] on the top surface (z = hz)
        x_samples = np.random.uniform(-hx/2, 0, size=n_points)
        y_samples = np.random.uniform( 0, hy/2, size=n_points)
        z_samples = np.random.uniform(hz + 0.2, hz + 0.4, size=n_points)  # top surface in local coordinates
        
        # Create an array of local points (in tray's coordinate system)
        points_local = np.vstack((x_samples, y_samples, z_samples)).T

        # Convert the tray's quaternion (PyBullet format: [x, y, z, w]) to a rotation matrix
        r = R.from_quat(tray_ori)
        # Transform the local points to world coordinates:
        # Apply rotation then translation.
        points_world = r.apply(points_local) + np.array(tray_pos)
        return points_world

    @staticmethod
    def get_pybullet_ee_ori(rai_orientation):
        """
        Convert a RAi configuration orientation quaternion (for the 'l_gripper' frame)
        in [w, x, y, z] order to a PyBullet end-effector orientation quaternion in [x, y, z, w] order.
        
        Args:
            rai_orientation: A quaternion from RAi in [w, x, y, z] order.
            
        Returns:
            A quaternion in PyBullet convention ([x, y, z, w]).
        """

        # Convert the RAi quaternion [w, x, y, z] to PyBullet format [x, y, z, w]
        q_pb_format = [rai_orientation[1], rai_orientation[2], rai_orientation[3], rai_orientation[0]]
        
        # Create a rotation object from this quaternion
        r_rai = R.from_quat(q_pb_format)
        
        # Define the transformation matrix T that was used in the forward conversion:
        # This matrix swaps the x and y axes and flips the z axis.
        T = np.array([[0, 1, 0],
                    [1, 0, 0],
                    [0, 0, -1]])
        T_rot = R.from_matrix(T)
        
        # Reverse the transformation:
        # In the forward function we had: r = r_pb * T_rot,
        # so to recover r_pb we compute: r_pb = r_rai * T_rot.inv()
        r_pb = r_rai * T_rot.inv()
        
        # Get the resulting quaternion in PyBullet convention ([x, y, z, w])
        q_pb = r_pb.as_quat()
        return q_pb

        
    def go_to_tray(self):
        """
            Sample possible goal position
            Move the EE to the goal
        """
        sampled_goals = self.goal_sampler()
        q = None
        
        for goal_position in sampled_goals:
            q = self.planner.compute_target_configuration(goal_position, target_ori=None)
            if q is not None:
                break
        if q is None:
            print("IK failed to compute a goal configuration after sampling.")
            return False
                
        self.planner.C.setJointState(q)
        l_gripper = self.planner.C.getFrame("l_gripper")
        position = l_gripper.getPosition()
        orientation_ry = l_gripper.getQuaternion()
        orientation = self.get_pybullet_ee_ori(orientation_ry)
        # self.planner.C.view(True)
        result = self.moveTo(position, orientation)

        return result
        
    def is_colliding(self, position):
        """
            Check if the robot is colliding with any obstacles
            Args:
                position: Desired end-effector position.
            Returns:
                True: if colliding
                False: if not colliding
        """
        # Check for collision
        obstacles = self.planner.get_obstacles()
        margin  = 0.05  # margin for collision detection
        for obstacle in obstacles:
            if obstacle is not None:
                obstacle_pos, obstacle_size = obstacle
                # Check if the robot is colliding with the obstacle
                distance = np.linalg.norm(position - obstacle_pos)
                if distance < obstacle_size + margin:
                    print("Collision detected with obstacle at position:", obstacle_pos)
                    return True
        return False

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

    def goTo(self, goal_position, goal_ori):
        """
            Does not use the planner, but directly moves to the goal position
            and orientation. This is used for the final goal position
            of the robot, where we do not want to check for collisions
            and just move to the goal position and orientation.
            Args:
                goal_position: Desired end-effector position.
                goal_ori: Desired end-effector orientation in PyBullet quaternion format.

            Returns:
                True: if goal pose possible and tries its best to EE
                False: if goal pose is not possible according to control module
        """
        qT = self.planner.compute_target_configuration(goal_position, goal_ori)
        if qT is None:
            print("IK failed to compute a goal configuration.")
            return False
        qT = np.array(qT)
        self.sim.robot.position_control(qT)
        MAX_ITER = 100
        epsilon = 0.03  # for position/config change
        for i in range(MAX_ITER):
            self.sim.step()
            robot_joint_config = self.sim.robot.get_joint_positions()
            delta_config = np.abs(robot_joint_config - qT)
            if (np.max(delta_config) < epsilon):
                print("Goal Achieved")
                break
        return True
    
    def sample_safe_config(self):
        """
            Sample safe configuration about the EE position
            Returns:
                q: joint configuration
        """

        ### sample points which are more likely to be feasible
        # heuristic is to sample points from cube which are closer to the Robot body
        # So, choosing (+z ,-z), -y, -x directions
        ee_pos, _ = self.sim.robot.get_ee_pose()
        size = 0.1
        MAX_CUBE_SIZE = 1.0
        while size < MAX_CUBE_SIZE:
            # Max attempts for current cube size
            MAX_ATTEMPTS = 10
            for i in range(MAX_ATTEMPTS):
                # Sample a biased offset vector within the cube with a bias:
                # For x and y: sample only negative offsets (closer to the robot body)
                offset_x = np.random.uniform(-size/2, 0)
                offset_y = np.random.uniform(-size/2, 0)
                
                # For z: allow both positive and negative offsets
                offset_z = np.random.uniform(-size/2, size/2)
                
                # Create the offset vector and compute the target EE position
                offset = np.array([offset_x, offset_y, offset_z])
                target_pos = ee_pos + offset
                
                q = None
                # Check if this possibly is colliding with any obstacles
                if not self.is_colliding(target_pos):
                    # Use inverse kinematics to obtain a joint configuration that reaches target_pos.
                    q = self.planner.compute_target_configuration(target_pos)
                
                if q is not None:
                    self.planner.C.setJointState(q)
                    l_gripper = self.planner.C.getFrame("l_gripper")
                    safe_pos, safe_ori = l_gripper.getPosition(), l_gripper.getQuaternion()
                    return q, safe_pos, safe_ori
            size = size * 1.2  # If no valid configuration found, increase the cube size
        return None
        

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
        replan_freq = 1
        MAX_ITER = 100
        next_q_threshold = 0.5
        epsilon = 0.03  # for position/config change
        check_new_config_freq = 10

        gp = self.planner

        # Get the target joint configuration
        qT = self.planner.compute_target_configuration( 
                                                        goal_position,
                                                        goal_ori
                                        )
        if qT is None:
            print("IK failed to compute a goal configuration.")
            return False
        qT = np.array(qT)

        last_configuration = self.sim.robot.get_joint_positions()
        last_pos, last_ori = self.sim.robot.get_ee_pose()
        fall_back_config = qT
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
                        # Trying multiple times beacause we have just grasped the object
                        # and probably out of obstacles' paths
                        MAX_ATTEMPTS = 10
                        while path is None:
                            for i in range(6):
                                self.sim.step()
                            path = gp.plan(goal_position, goal_ori)
                            MAX_ATTEMPTS -= 1
                            if MAX_ATTEMPTS == 0:
                                break
                        if path is None:
                            print("No path found")
                            return False
                if path is not None:
                    initial_q = path[0]
                    q = path[-1]
                    fall_back_config = path[-1]
                    for next_q in path[1:]:
                        if np.linalg.norm(initial_q - next_q) > next_q_threshold:
                            q = next_q
                            break
                if path is None:
                    q = fall_back_config
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