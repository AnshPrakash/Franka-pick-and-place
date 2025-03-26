# Tracking Obstacles

This module estimates the 3D location, i.e., the 6D pose of an obstacle in the simulation.


# Idea:
1. Estimate (x,y,z) position of the obstacles using depth image, segmentation mask and RGB image
2. Create a Kalman Filter for the dynamics of the obstacles
3. Integrate Kalman Filter with Estimates of step 1




# Dependencies

```
pip install filterpy
```

# State Transition Model

```
State Transition Model
For a constant-velocity model, the discretized equations (using a time step dt) are:

  xₖ₊₁ = xₖ + vx·dt
  vxₖ₊₁ = vxₖ
  yₖ₊₁ = yₖ + vy·dt
  vyₖ₊₁ = vyₖ
  zₖ₊₁ = zₖ + vz·dt
  vzₖ₊₁ = vzₖ
```

# Least Squares to Estimate the Center of obstacles

```
Use point clouds and formulate a Least Squares Estimation problem

```