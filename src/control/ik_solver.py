from typing import Dict, Any, Optional

import numpy as np
import pybullet as p

# from src.robot import Robot
# from src.simulation import Simulation
# from src.utils import pb_image_to_numpy

import robotic as ry
import numpy as np
import time

C = ry.Config()
C.addFile(ry.raiPath('scenarios/pandaSingle.g'))
C.view(True)

C.addFrame('box') \
    .setPosition([-.25,.1,1.]) \
    .setShape(ry.ST.ssBox, size=[.06,.06,.06,.005]) \
    .setColor([1,.5,0]) \
    .setContact(1)
C.view(True)


class IKSolver:
    """IKSolver Class.

    The class initializes
    """
    pass