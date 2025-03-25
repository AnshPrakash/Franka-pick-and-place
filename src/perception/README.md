# Strategy
- Extract pointclouds from depth image by inverse transformation with view and projection matrix
- Get GT mesh for every object, apply scaling and sample a reference pointcloud
- Target object scan: use defined EE poses to take multiple depth images, extract pointclouds and merge them
## Perception Algorithm
- RANSAC as global registration algorithm
  - Compute useful features for RANSAC
  - downsample pointclouds
- ICP as local registration algorithm
  - Use RANSAC output as initial transformation
  - refine with ICP
- Repeat Registration multiple times in case of failure

# Additional Packages
- open3d
- trimesh