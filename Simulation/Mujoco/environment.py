import mujoco as mj
import mujoco.viewer
import numpy as np
import time
from dm_control import mujoco
from dm_control.utils import inverse_kinematics

class Env:
    def __init__(self, path="C:/Users/dexte/Documents/mujoco_menagerie-main/kuka_iiwa_14/", timestep=1/240.,realtime=False,speed=1):
        self.realtime=realtime
        self.timestep=timestep
        self.reset()
        self.speed=speed
        self.base_xml="""<mujoco model="iiwa14 scene">
  <include file=\""""+path+"""iiwa14.xml"/>

  <statistic center="0.2 0 0.2" extent="1.0"/>

  <visual>
    <headlight diffuse="0.6 0.6 0.6" ambient="0.3 0.3 0.3" specular="0 0 0"/>
    <rgba haze="0.15 0.25 0.35 1"/>
    <global azimuth="-120" elevation="-20"/>
  </visual>

  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7" rgb2="0 0 0" width="512" height="3072"/>
    <texture type="2d" name="groundplane" builtin="checker" mark="edge" rgb1="0.2 0.3 0.4" rgb2="0.1 0.2 0.3"
      markrgb="0.8 0.8 0.8" width="300" height="300"/>
    <material name="groundplane" texture="groundplane" texuniform="true" texrepeat="5 5" reflectance="0.2"/>
  </asset>

  <worldbody>
    <light pos="0 0 1.5" dir="0 0 -1" directional="true"/>
    <geom name="floor" size="0 0 0.05" type="plane" material="groundplane"/>
  </worldbody>
</mujoco>
"""
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
    def update_task(self):
        self.model = mj.MjModel.from_xml_string(self.base_xml)
        self.data = mj.MjData(self.model)
        self.physics = mujoco.Physics.from_xml_string(self.base_xml)
    def generate(self): #after all the selections are made this generates the simulation
        #self.model = mujoco.MjModel.from_xml_path(path+"iiwa14.xml")
        #self.data = mujoco.MjData(self.model)
        pass 
    def generate_block_xml(name, pos, size=(0.02, 0.02, 0.02), color=(1, 0, 0, 1)):
        return f"""
        <body name="{name}" pos="{pos[0]} {pos[1]} {pos[2]}">
            <geom type="box" size="{size[0]} {size[1]} {size[2]}" rgba="{color[0]} {color[1]} {color[2]} {color[3]}"/>
        </body>
        """
    def generate_blocks(self,num_blocks):
        xml = ""
        for i in range(num_blocks):
            xml += f'''
            <body name="block_{i}" pos="{i*0.2} 0 0.02">
                <geom type="box" size="0.02 0.02 0.02"/>
            </body>
            '''
        #add to environment
        insert_point = self.base_xml.find("</worldbody>")

        block_xml = xml
        self.base_xml=self.base_xml[:insert_point] + block_xml + "\n" + self.base_xml[insert_point:]
        print(self.base_xml)
    def step(self):
        pass
    def pick_block(self, block_id):
        pass 
    def put_block(self):
        pass 
    def move_gripper_to(self, fingertip_coords,vel=0.9):
        result = inverse_kinematics.qpos_from_site_pose(
            self.physics,
            site_name="attachment_site",   # must exist in your XML
            target_pos=fingertip_coords,
            max_steps=100
        )
        self.physics.data.qpos[:] = result.qpos
        self.physics.forward()
        self.data.qpos[:] = self.physics.data.qpos[:]
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
    e.update_task()
    viewer=mj.viewer.launch_passive(e.model, e.data)
    viewer.close()
    e.generate_blocks(5)
    e.update_task()
    viewer=mj.viewer.launch_passive(e.model, e.data)
    t = 0
    while viewer.is_running() and t<100:
            #self.data.ctrl[0] = 1.0 * np.sin(t)
            mj.mj_step(e.model, e.data)
            e.move_gripper_to([0,0.4,0.2])
            viewer.sync()
            t += 0.01
    viewer.close()