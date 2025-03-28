import numpy as np
import robotic as ry
import pybullet as p

from src.simulation import Simulation
from src.control import IKSolver
from src.tracking import Track


class Global_planner(IKSolver):
    def __init__(self, sim: Simulation):
        """
            Initialize KOMO planner with robot configuration.
            Estimate the robot's joint configuration given the target end-effector position 
            and orientation.
        """
        super().__init__(sim)
        
        self.obstacles_tracker = []
        for obstacle in self.sim.obstacles:
            self.obstacles_tracker.append(Track(obstacle.id,sim))

    def get_obstacles(self) -> list:
        """
            Get the center of obstacle centers and their radii
            Output: List[(position, size)]
        """
        obstacles = []
        for tracker in self.obstacles_tracker:
            obstacles.append((tracker.estimate_position(), tracker.object_size))
        return obstacles


    def plan(self, target_pos, target_ori):
        """
        Compute a collision-free path to the goal position using RRT-based planning,
        taking into account two spherical obstacles.
        
        Args:
            target_pos: Desired end-effector position.
            target_ori: Desired end-effector orientation in PyBullet quaternion format.
        
        Returns:
            A numpy array with the planned path (a sequence of joint configurations), or
            None if planning fails.
        """
        # 1. Compute the goal configuration (qT) using IK.
        qT = self.compute_target_configuration(target_pos, target_ori)
        if qT is None:
            print("IK failed to compute a goal configuration.")
            return None

        # 2. Get the current robot configuration (q0)
        q0 = self.C.getJointState()
        print("Start configuration (q0):", q0)
        print("Goal configuration (qT):", qT)

        # 3. Add obstacles to the configuration as proper collision objects.
        #    get_obstacles() returns a list of (position, radius) tuples.
        obstacles = self.get_obstacles()
        obstacle_names = []
        for i, (pos, radius) in enumerate(obstacles):
            obs_name = f"obstacle_{i}"
            # Check if an obstacle frame already exists; if not, add it.
            obs = self.C.getFrame(obs_name)
            if obs is None:
                obs = self.C.addFrame(obs_name)
            # Use a sphere shape (not a marker) so it can be used for collision checking.
            obs.setShape(ry.ST.sphere, [radius])
            obs.setPosition(pos)
            obs.setContact(1.0)
            obs.setColor([1.0, 0.0, 0.0])  # Red for obstacles.
            obstacle_names.append(obs_name)

        # 4. Create an instance of the RRT planner.
        rrt = ry.RRT_PathFinder()
        # Set the planning problem using the current configuration.
        rrt.setProblem(self.C)
        # Set the start and goal configurations.
        rrt.setStartGoal([q0], [qT])
        
        # 5. Explicitly add collision pairs: make sure that the left gripper and each obstacle are checked.
        collision_pairs = []
        for obs_name in obstacle_names:
            # Each collision pair is specified as two frame names.
            collision_pairs.extend(["l_gripper", obs_name])
        if collision_pairs:
            rrt.setExplicitCollisionPairs(collision_pairs)
        
        # 6. Solve the planning problem.
        ret = rrt.solve()
        if not ret.feasible:
            print("Path planning returned an infeasible solution.")
            return None

        # 7. Retrieve the resampled path.
        path = ret.x
        print("Path found with", path.shape[0], "configurations.")

        # Visualize the planned path.
        # import time
        # for t in range(path.shape[0]):
        #     self.C.setJointState(path[t])
        #     self.C.view(False, f'Path slice {t}')
        #     time.sleep(0.1)
        # print("Return Path", len(path))
        return path
