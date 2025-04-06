import numpy as np
import pybullet as p
import robotic as ry
from scipy.spatial.transform import Rotation as R

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
    

    def goal_sampler(self):
        """
            Return :
                Sampled goal position
                Sampled goal orientation
        """
        tray_pos, tray_ori = p.getBasePositionAndOrientation(self.sim.goal.id)
        tray_id = self.sim.goal.id
        collision_data = p.getCollisionShapeData(tray_id, -1)
        tray_size = collision_data[0][3]  # Extracting halfExtents (for box shapes)
        tray_size = np.array(tray_size)
        hx, hy, hz = tray_size  # half extents along x, y, and z.
        margin = 0.05
        # avoid points close to border
        hx = hx - margin
        hy = hy - margin
        n_points = 200

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

        sampled_goals = points_world
        
        for goal_position in sampled_goals:
            q = self.planner.compute_target_configuration(goal_position, target_ori=None)
            if q is not None:
                break
        if q is None:
            print("IK failed to compute a goal configuration after sampling.")
            return None, None
                
        self.planner.C.setJointState(q)
        l_gripper = self.planner.C.getFrame("l_gripper")
        position = l_gripper.getPosition()
        orientation_ry = l_gripper.getQuaternion()
        orientation = self.get_pybullet_ee_ori(orientation_ry)

        return position, orientation

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
        position, orientation = self.goal_sampler()
        while position is None:
            print("No goal position found: Retrying...")
            position, orientation = self.goal_sampler()

        result = self.moveTo(position, orientation)

        return result
        
    def is_colliding(self, joint_config = None, position = None):
        """
            Check if the robot is colliding with any obstacles
            Args:
                position: Desired end-effector position.
            Returns:
                True: if colliding
                False: if not colliding
        """
        if position is None:
            if joint_config is None:
                raise ValueError("Either joint_config or position must be provided.")
            self.planner.C.setJointState(joint_config)
            l_gripper = self.planner.C.getFrame("l_gripper")
            position = l_gripper.getPosition()
        else:
            q = self.planner.compute_target_configuration(position)
            if q is None:
                # infeasible configuration
                return True
            joint_config = q
            self.planner.C.setJointState(joint_config)
        # Check for collision
        # obstacles = self.planner.get_obstacles()
        margin  = 0.02  # margin for collision detection
        
        
        # update obstacle position
        obstacles = self.planner.get_obstacles()
        obstacle_names = []
        C = self.planner.C
        for i, (pos, radius) in enumerate(obstacles):
            obs_name = f"obstacle_{i}"
            # Check if an obstacle frame already exists; if not, add it.
            obs = C.getFrame(obs_name)
            if obs is None:
                obs = C.addFrame(obs_name)
            # Use a sphere shape (not a marker) so it can be used for collision checking.
            obs.setShape(ry.ST.sphere, [radius])
            obs.setPosition(pos)
            obs.setContact(1)
            obs.setColor([1.0, 0.0, 0.0])  # Red for obstacles.
            obstacle_names.append(obs_name)

        # Check collision with the whole Robot
        for obs_name in obstacle_names:
            for robot_frame_name in self.planner.robot_frames:
                d, _ = C.eval(ry.FS.negDistance, [robot_frame_name, obs_name])
                if np.abs(d) < margin:
                    print(f"Distance between {robot_frame_name} and {obs_name}: {d}")
                    # C.computeCollisions()
                    # penetration, _ = C.eval(ry.FS.accumulatedCollisions, [])
                    # print("Collisions", C.getCollisions())
                    # info = C.eval(ry.FS.accumulatedCollisions, [])
                    # print("Info", info )
                    # C.view(True)
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
    
    def sample_safe_config_radial(self):
        """
        Sample a safe configuration by retracting the end-effector (EE) 
        horizontally toward the robot base while keeping the EE height nearly constant 
        (with a small random variation).
        
        Returns:
            q: joint configuration,
            safe_pos: final EE position,
            safe_ori: final EE orientation
        """

        # Retrieve current end-effector and base poses
        ee_pos, _ = self.sim.robot.get_ee_pose()      # ee_pos: [x, y, z]
        base_pos = self.sim.robot.pos   # base_pos: [x, y, z]
        
        # Project positions onto horizontal plane (x-y) and compute radial vector
        ee_xy = np.array([ee_pos[0], ee_pos[1]])
        base_xy = np.array([base_pos[0], base_pos[1]])
        radial_vector_xy = ee_xy - base_xy
        norm_xy = np.linalg.norm(radial_vector_xy)
        if norm_xy == 0:
            return None, None, None  # avoid division by zero
        rad_dir_xy = radial_vector_xy / norm_xy

        # Define maximum retraction distance (e.g., up to 70% of the current horizontal distance)
        max_retraction = 0.7 * norm_xy
        MAX_ATTEMPTS = 20

        for i in range(MAX_ATTEMPTS):
            # Sample a random retraction distance along the horizontal direction
            d = np.random.uniform(0, max_retraction)
            # Compute new horizontal (x,y) position by retracting along the radial direction
            target_xy = ee_xy - d * rad_dir_xy
            
            # For the z coordinate, keep it almost the same with a small random variation 
            z_variation = np.random.uniform(-0.15, 0.15)
            target_z = ee_pos[2] + z_variation
            
            # Form the new target position
            target_pos = np.array([target_xy[0], target_xy[1], target_z])
            
            # Check for collisions at the target position
            q = None
            if not self.is_colliding(position=target_pos):
                # Compute the candidate joint configuration using inverse kinematics
                q = self.planner.compute_target_configuration(target_pos)
            
            if q is not None:
                self.planner.C.setJointState(q)
                l_gripper = self.planner.C.getFrame("l_gripper")
                safe_pos, safe_ori_ry = l_gripper.getPosition(), l_gripper.getQuaternion()
                safe_ori = self.get_pybullet_ee_ori(safe_ori_ry)
                return q, safe_pos, safe_ori

        return None, None, None



    def sample_safe_config(self):
        """
            Sample safe configuration about the EE position
            Returns:
                q: joint configuration
        """

        ### sample points which are more likely to be feasible
        # heuristic is to sample points from a cube around EE
        ee_pos, _ = self.sim.robot.get_ee_pose()
        size = 0.5
        MAX_CUBE_SIZE = 1.0
        while size < MAX_CUBE_SIZE:
            # Max attempts for current cube size
            MAX_ATTEMPTS = 10
            for i in range(MAX_ATTEMPTS):
                # Sample a biased offset vector within the cube with a bias:
                # For x and y: sample only negative offsets (closer to the robot body)
                offset_x = np.random.uniform(-size/2, size/2)
                offset_y = np.random.uniform(-size/2, -size/2)
                
                # For z: allow both positive and negative offsets
                offset_z = np.random.uniform(-size/2, size/2)
                
                # Create the offset vector and compute the target EE position
                offset = np.array([offset_x, offset_y, offset_z])
                target_pos = ee_pos + offset
                
                # self.planner.C.view(True)
                q = None
                # Check if this possibly is colliding with any obstacles
                if not self.is_colliding(position=target_pos):
                    # Use inverse kinematics to obtain a joint configuration that reaches target_pos.
                    q = self.planner.compute_target_configuration(target_pos)
                
                if q is not None:
                    self.planner.C.setJointState(q)
                    l_gripper = self.planner.C.getFrame("l_gripper")
                    safe_pos, safe_ori_ry = l_gripper.getPosition(), l_gripper.getQuaternion()
                    safe_ori = self.get_pybullet_ee_ori(safe_ori_ry)                    
                    return q, safe_pos, safe_ori
            size = size * 1.2  # If no valid configuration found, increase the cube size
        return None, None, None
        
    def backoff_strategy(self):
        """
            Move to a safer configuration to avoid collision
            Returns:
                q: joint configuration
        """
        # Sample a safe configuration
        q, safe_pos, safe_ori = self.sample_safe_config_radial()
        if q is None:
            print("No safe configuration found")
            return None, None, None
        else:
            # Move to the safe configuration and hope EE doesn't collide
            # with any obstacles
            self.goTo(safe_pos, safe_ori)
        MAX_WAIT_STEPS = 1000
        retry_freq = 10
        for i in range(1,MAX_WAIT_STEPS + 1):
            # dynamically check if the current robot configuration is safe
            # update to new safe state
            if self.is_colliding(position=safe_pos):
                # Sample a new safe configuration
                q, safe_pos, safe_ori = self.sample_safe_config_radial()
                if q is None:
                    print("No safe configuration found")
                    return None, None, None
                else:
                    # Move to the safe configuration and hope EE doesn't collide
                    # with any obstacles
                    self.goTo(safe_pos, safe_ori)
            

            # sample new goal position
            if (i%retry_freq == 0):
                goal_position, goal_orientation = self.goal_sampler()
            
                path = self.planner.plan(goal_position, goal_orientation)
                # safe_path = False
                if path is not None:
                    return path, goal_position, goal_orientation
                #     for q in path:
                #         if self.is_colliding(joint_config=q):
                #             safe_path = True
                #             break
                # if safe_path:
                #     return path, goal_position, goal_orientation
            
            # for faster simulation 
            # trade-off with no collision-checking
            for _ in range(4):
                self.sim.step()        
        return None, None, None


            

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
        replan_freq = 2
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
                if self.is_colliding(joint_config=q):
                    # Bring robot to a safe configuration
                    # set a new goal and provides a replanned path
                    print("We can collide Back off strategy is on")
                    new_path, new_goal_position, new_goal_ori = self.backoff_strategy() # move to a safer configuration to avoid collision
                    q = None
                    if new_path is not None:
                        # Move to the new goal position
                        # TODO: refactor this repeated code
                        initial_q = new_path[0]
                        q = new_path[-1]
                        fall_back_config = new_path[-1]
                        goal_position = new_goal_position
                        goal_ori = new_goal_ori
                        for next_q in path[1:]:
                            if np.linalg.norm(initial_q - next_q) > next_q_threshold:
                                q = next_q
                                break
                    print("Recovery complete from backoff move towards goal again--")
                    if q is None:
                        # no safe configuration found 
                        # Just keep moving forward even if it kills you 
                        print("No safe configuration found -- Hope for the best")
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