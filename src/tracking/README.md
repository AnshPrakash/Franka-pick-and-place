# Tracking Obstacles

This module estimates the 3D location, i.e., the 6D pose of an obstacle in the simulation.


# Idea:
1. Estimate (x,y,z) position of the obstacles using depth image, segmentation mask and RGB image
2. Create a Kalman Filter for the dynamics of the obstacles
3. Integrate Kalman Filter with Estimates of step 1




