# Additional Dependencies
- pip install torch==2.1.0 torchvision==0.16.0 torch-scatter==2.1.2 -f https://data.pyg.org/whl/torch-2.1.0+cu121.html

- git clone -b ipynb https://github.com/iROSA-lab/GIGA.git
- pip install -r GIGA/requirements.txt
- cd GIGA && pip install -e .

- Note, cloning not needed, because GIGA already in repository
# Comments
- might also push whole GIGA repository as changes to some source files
- cause were deprecated numpy syntax (np.bool, np.int) but other packages require newer versions
- chnages in `GIGA/src/giga/ConvONets/utils/binvox_rw.py`

# Report
- In general: Based on the retrieved pose from perception, we use the GT mesh for the object and transform it accordingly
- Implicit Grasping
  - Have to convert GT mesh to TSDF function and evaluate in a defined grid
  - GIGA model never outputs a grasp
  - Would require much more debugging: Could be TSDF quality, the grid where it is evaluated on (including scaling of object), model quality etc.
- Sampling based
  - Works well with a sampled Pointcloud based on our GT mesh
  - Benefits from adjusting parameters
  - Requires scaling for some objects or no grasps outputted