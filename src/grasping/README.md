# Additional Dependencies
- pip install torch==2.1.0 torchvision==0.16.0 torch-scatter==2.1.2 -f https://data.pyg.org/whl/torch-2.1.0+cu121.html

- git clone -b ipynb https://github.com/iROSA-lab/GIGA.git
- pip install -r GIGA/requirements.txt
- cd GIGA && pip install -e .

# Comments
- might also push whole GIGA repository as changes to some source files
- cause were deprecated numpy syntax (np.bool, np.int) but other packages require newer versions
- chnages in `GIGA/src/giga/ConvONets/utils/binvox_rw.py`