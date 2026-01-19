import mujoco
import mujoco.viewer
import numpy as np
import time
path="/its/home/drs25/Documents/mujoco_menagerie-main/kuka_iiwa_14/"
class Env:
    def __init__(self, timestep=1/240.,realtime=False,speed=1):
        self.realtime=realtime
        self.timestep=timestep
        self.reset()
        self.speed=speed
    def reset(self): #reset simulation variables
        self.block_file=[]
        self.positions=[]
        self.colours=[]
        self.sizes=[]
        self.block_ids = []
        self.physics=[]
        self.ee_index = 6  # End effector link
        self.gripper_joints = [7, 8]
        self.arm_joints = list(range(7))
        self.holding_constraint = None
        # Block tracking
        self.fingertip_coords=[]
        self.recording = False
        self.video_id = None
        self.timestep = self.timestep
        self.move_Step_amount=0.2
        self.realtime=self.realtime
    def generate(self): #after all the selections are made this generates the simulation
        #self.model = mujoco.MjModel.from_xml_path(path+"iiwa14.xml")
        #self.data = mujoco.MjData(self.model)
        pass 
    def step(self):
        pass
    def generate_blocks(self,num):
        pass 
    def generate_block(self,position,colour,size=1,shape="cube"):
        pass
    def pick_block(self, block_id):
        pass 
    def put_block(self):
        pass 
    def move_gripper_to(self, fingertip_coords, euler=[0, 3.14, 0],vel=0.9):
        pass 
    def get_observation(self):
        pass 
    def close(self):
        pass 
    def recreate_from_file(self,env):
        #=load in a file and recreate the objects where they should be
        self.reset()
        temp=self.realtime 
        temp2=self.timestep
        self.__dict__.update(env.__dict__)
        self.realtime=temp 
        self.timestep=temp2
        self.populate()
    def populate(self):
        pass 
    def move_up(self):
        robot_coords=list(p.getLinkState(self.robot_id, linkIndex=self.ee_index)[0])
        robot_coords[2]+=self.move_Step_amount
        self.move_gripper_to(robot_coords)
    def move_down(self):
        robot_coords=list(p.getLinkState(self.robot_id, linkIndex=self.ee_index)[0])
        robot_coords[2]-=self.move_Step_amount
        self.move_gripper_to(robot_coords)
    def left(self):
        robot_coords=list(p.getLinkState(self.robot_id, linkIndex=self.ee_index)[0])
        robot_coords[1]+=self.move_Step_amount
        self.move_gripper_to(robot_coords)
    def right(self):
        robot_coords=list(p.getLinkState(self.robot_id, linkIndex=self.ee_index)[0])
        robot_coords[1]-=self.move_Step_amount
        self.move_gripper_to(robot_coords)
    def forward(self):
        robot_coords=list(p.getLinkState(self.robot_id, linkIndex=self.ee_index)[0])
        robot_coords[0]+=self.move_Step_amount
        self.move_gripper_to(robot_coords)
    def backward(self):
        robot_coords=list(p.getLinkState(self.robot_id, linkIndex=self.ee_index)[0])
        robot_coords[0]-=self.move_Step_amount
        self.move_gripper_to(robot_coords)




if __name__=="__main__":
    e=Env()
    viewer=mujoco.viewer.launch_passive(e.model, e.data)
    t = 0
    while viewer.is_running() and t<100:
            #self.data.ctrl[0] = 1.0 * np.sin(t)
            mujoco.mj_step(e.model, e.data)
            viewer.sync()
            t += 0.01
    viewer.close()