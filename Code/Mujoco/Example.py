import numpy as np 
import matplotlib.pyplot as plt
import sys
sys.path.append("../..")
import os
PROJECT_ROOT = r"C:/Users/dexte/Documents/GitHub/Robot_shape_learning"
sys.path.append(PROJECT_ROOT)
from Simulation.Mujoco.environment import *

e=Env(path="C:/Users/dexte/Documents/GitHub/Robot_shape_learning/Assets/kuka_iiwa_14/",realtime=1,speed=1/440)
e.generate_blocks(5)
e.update_task()
viewer= mj.viewer.launch_passive(e.model, e.data)
for i in range(5):
    #self.data.ctrl[0] = 1.0 * np.sin(t)
    e.move_gripper_to([0.4+(i*0.1),0.4,0.1])
    e.step(viewer=viewer)
    e.pick_block()
    viewer.sync()
    e.move_gripper_to([0.4,0.4,0.5])
    e.step(viewer=viewer)
    viewer.sync()
    e.move_gripper_to([0.4,-0.4,0.3])
    e.step(viewer=viewer)
    viewer.sync()
    e.put_block()
    e.step(viewer=viewer)
    viewer.sync()
    e.move_gripper_to([0.5,0.4,0.5])
    e.step(viewer=viewer)
    viewer.sync()
viewer.close()