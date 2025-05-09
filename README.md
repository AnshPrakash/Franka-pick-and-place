# Intelligent Robotic Manipulation

The overall Goal of the project is to grasp a YCB-Object and place it in a goal basket while avoiding obstacles. To do
this we have provided some helpful tips/information on various subtasks that you may or may not need to solve. Perception (task 1) to detect the graspable object, Controller (task 2) to move your robot arm, sample and execute a grasp (task 3), localize and track obstacles (task 4) and plan the trajectory to place the object in the goal, while avoiding the obstacles (task 5). Note that, what we mention in the task subparts are just for guidance and you are fully free to choose whatever you want to use to accomplish the full task. But you need to make sure that you don't use privilege information from the sim in the process.

We highly recommend you go through the [Pybullet Documentation](https://pybullet.org/wordpress/index.php/forum-2/)

Make sure you have [anaconda](https://www.anaconda.com/) or [miniconda](https://docs.conda.io/projects/miniconda/en/latest/miniconda-install.html) installed beforehand.

```shell
git clone https://github.com/iROSA-lab/irobman_project_wise2324.git
cd irobman_project_wise2324
conda env create --name irobman -f environment.yml
conda activate irobman
git clone https://github.com/eleramp/pybullet-object-models.git # inside the irobman_project folder
pip install -e pybullet-object-models/
pip install torch==2.1.0 torchvision==0.16.0 torch-scatter==2.1.2 -f https://data.pyg.org/whl/torch-2.1.0+cu121.html # 11.8 wheels as in repo will lead to conflicts
pip install -r GIGA/requirements.txt
cd GIGA && pip install -e .
sudo apt install liblapack3 freeglut3 libglew-dev # control system installations
```

## Important

```
To set the joint limits to pybullet's expected values
Copy the file from src/control/panda_model/panda_irobman.g to ry.raiPath('scenarios/panda_irobman.g')
```

Note that you should check after the installation if pybullet is using numpy or not by running `pybullet.isNumpyEnabled()` in your code. Everything can still run without this too but it will be slow. You can also increase the speed of execution by choosing not to see the camera output in the GUI by toggling `cam_render_flag`. You will be able to still see the GUI but the cam output there will not be visible.

## Experiments
Our experiments can be recreated using the `launch.py` script. For our results we used the following runs:
```shell
python launch.py --all
python launch.py --all --obstacles --steps 5
python launch.py --object "YcbHammer" --obstacles
```

## Codebase Structure

```shell

├── configs
│   └── test_config.yaml # config file for your experiments (you can make your own)
├── main.py # example runner file (you can add a bash script here as well)
├── README.md
└── src
    ├── objects.py # contains all objects and obstacle definitions
    ├── robot.py # robot class
    ├── simulation.py # simulation class
    └── utils.py # helpful utils

```

### Things you can change and data that you can use:

- Testing with custom objects
- Testing with different control modes
- Toggling obstacles on and off
- Adding new metrics for better report and explanation
- Information such as:
    - Robot joint information
    - Robot gripper and end-effector information
    - Any information from camera
    - Goal receptacle position
    - Camera position/matrices

### Things you cannot change without an explicit request to course TA's/Professor:
- Any random sampling in the sim
- Number of obstacles
- Ground truth position of the object in question
- Any ground truth orientation
- Robot initial position and arm orientation
- Goal receptacle position
- Using built in high level pybullet methods like `calculateInverseDynamics`, `calculateInverseKinematics`


_Note: If you want to add a new metric you can use ground truth information there but only for comparison with your prediction._

### Checkpoints & Marks:
- Code Related
    - Being able to detect the object and get it’s pose (+15)
    - Moving the arm to the object (+15)
    - Being able to grasp object (+15)
    - Being able to move the arm with the object to goal position (without obstacles) (+20)
    - Detecting and tracking the obstacles (+15)
    - Full setup: Being able to execute pick and place with obstacles present (+30)
- A part of your marks is also fixed on the report (+10)

We will only consider the checkpoints as complete if you provide a metric or a success rate for each. 
The format of the report will be the standard TU-Darmstadt format.

### Submission format:
- Link to your github/gitlab repository containing a well documented README with scripts to run and test various parts of the system.
- Report PDF.

## Task 1 (Perception)
Implement an object 6D pose estimation workflow using a simulation environment with two cameras. The first camera is static, positioned in front of a table, and is suitable for obstacle detection and coarse object localization. The second camera is mounted on the robot’s end-effector, aligned with robot link 11, and is used for refined pose estimation and grasping. Note that the camera's Y-axis points upward. For debugging, you can press W to toggle wireframe mode and J to display the axes in the GUI. You are encouraged to integrate state-of-the-art (SOTA) model-based 6D pose estimation methods or apply conventional approaches such as RANSAC and Iterative Closest Point (ICP), as demonstrated in the course. Additionally, synthetic masks generated by PyBullet can be used for object segmentation. 

*Reference*
Global registration (coarse pose estimation) [Tutorial](https://www.open3d.org/docs/release/tutorial/pipelines/global_registration.html)
ICP registration (coarse pose estimation) [Tutorial](https://www.open3d.org/docs/release/tutorial/pipelines/global_registration.html)
MegaPose, a method to estimate the 6D pose of novel objects, that is, objects unseen during training. [repo](https://github.com/megapose6d/megapose6d)


## Task 2 (Control)

Implement an IK-solver for the Franka-robot. You can use the pseudo-inverse or the transpose based solution. Use Your IK-solver to move the robot to a certain goal position. This Controller gets used throughout the project (e.g. executing the grasp - moving the object to the goal).

## Task 3 (Grasping)

Now that you have implemented a controller (with your IK solver) and tested it properly, it is time to put that to good use. From picking up objects to placing them a good well-placed grasp is essential. Hence, given an object you have to design a system that can effectively grasp it. You can use the model from the ![Grasping exercise](https://github.com/iROSA-lab/GIGA) and ![colab](https://colab.research.google.com/drive/1P80GRK0uQkFgDbHzLjwahyJOalW4M5vU?usp=sharing) to sample a grasp from a point-cloud. We have added a camera, where you can specify its position. You can set the YCB object to a fixed one (e.g. a Banana) for development. Showcase your ability to grasp random objects
for the final submission.

## Task 4 (Localization & Tracking)

After you have grasped the object you want to place it in the goal-basket. In order to avoid the obstacles (red spheres), you need to track them. Use the provided fixed camera and your custom-positioned cameras as sensors to locate and track the obstacles. Visualize your tracking capabilities in the Report (optional) and use this information to avoid collision with them in the last task. You could use a Kalman Filter (e.g. from Assignment 2).

## Task 5 (Planning)

After you have grasped the YCB object and localized the obstacle, the final task is to plan the robot’s movement in order to place the object in the goal basket. Implement a motion planner to avoid static and dynamic obstacles and execute it with your controller. Once you are above the goal-basket open the gripper to drop the object in the goal.

The most naive/cheating way of doing grasping can be run as
```
python3 -m pybullet_robots.panda.loadpanda_grasp
```
The code can be found [here](https://github.com/bulletphysics/bullet3/blob/master/examples/pybullet/gym/pybullet_robots/panda/loadpanda_grasp.py), and there is no collision avoidance at all.

Motion planning can be generally decoupled into global planning and motion planning. Global planning is responsible for generate a reference path to avoid static obstacles while local planning is for keeping track of the reference path and avoid dynamic obstacles. 
* For global planning, 
    * An easy way is to use sampling-based methods (rrt, prm) to sample the reference path directly in 3d space and use your designed IK solver to track the path, see [here](https://github.com/yijiangh/pybullet_planning/tree/dev/src/pybullet_planning/motion_planners) for more algorithmic details.
    * A faster but difficult solution is to sample the path directly in the configuration space so here you do not need the IK solver. You can see [here](https://github.com/sea-bass/pyroboplan) for an example, though it is not implemented using pybullet.
* For local planning, after you have a prediction of the moving obstacles,
    * An easy way is to use potential field method to avoid them. You can check [here](https://github.com/PulkitRustagi/Potential-Field-Path-Planning) for more details.
    * A more advanced approach is to use MPC or sampling-based MPC to handle the moving obstacles. You can check [this](https://github.com/tud-amr/m3p2i-aip) for more details, though the kinematic model and collision checking are done in IsaacGym.

You can decide to use whichever techniques to solve the task, you can use either a `global planner` and a `local planner` combination, or you can directly use the sampling-based MPC to avoid static and dynamic obstacles.

# Final Words:
We hope you have fun and explore robotics more deeply through this project.
