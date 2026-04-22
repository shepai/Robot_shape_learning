import mujoco as mj
import mujoco.viewer
import numpy as np
import time
from dm_control import mujoco
from dm_control.utils import inverse_kinematics

class Env:
    def __init__(self, path="C:/Users/dexte/Documents/mujoco_menagerie-main/kuka_iiwa_14/", timestep=1/240.,realtime=False,speed=1/24):
        self.realtime=realtime
        self.attached_block=None
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
        self.base_xml_clone=self.base_xml
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
        block_xml = ""
        weld_xml = ""
        for i in range(num_blocks):
            block_xml += f'''
            <body name="block_{i}" pos="{(i*0.08)+0.4} 0.4 0.05">
                <joint type="free"/>
                <geom type="box" size="0.02 0.02 0.02" mass="1"/>
                <site name="site_{i}" pos="0 0 0.1"/>
            </body>
            '''

            # 🔥 one weld per block (inactive by default)
            weld_xml += f'''
            <weld name="weld_block_{i}" site1="attachment_site" site2="site_{i}" active="false"/>
            '''

        insert_point = self.base_xml_clone.find("</worldbody>")
        self.base_xml = (
            self.base_xml_clone[:insert_point] +
            block_xml +
            "\n" +
            self.base_xml_clone[insert_point:]
        )

        # --- insert (or create) equality section ---
        if "<equality>" in self.base_xml:
            insert_eq = self.base_xml.find("</equality>")
            self.base_xml = (
                self.base_xml[:insert_eq] +
                weld_xml +
                "\n" +
                self.base_xml[insert_eq:]
            )
        else:
            # create equality section if it doesn't exist
            insert_point = self.base_xml.find("</mujoco>")
            eq_section = f"<equality>\n{weld_xml}\n</equality>\n"
            self.base_xml = (
                self.base_xml[:insert_point] +
                eq_section +
                self.base_xml[insert_point:]
            )
    def pwm(self, kp=200, kd=1): 
        q = self.data.qpos[:7]
        qd = self.data.qvel[:7]

        torque = kp * (self.targets - q) - kd * qd
        self.data.ctrl[:7] = torque
    def step(self, step_size=100, viewer=None):
        site_id = self.model.site("attachment_site").id

        for _ in range(step_size):
            self.pwm()
            mj.mj_step(self.model, self.data)
            if self.attached_block is not None:
                site_id = self.model.site("attachment_site").id
                block_bid = self.model.body(self.attached_block).id

                joint_id = self.model.body_jntadr[block_bid]
                qpos_adr = self.model.jnt_qposadr[joint_id]
                qvel_adr = self.model.jnt_dofadr[joint_id]

                self.data.qpos[qpos_adr:qpos_adr+3] = self.data.site_xpos[site_id]
                quat = np.zeros(4)
                mj.mju_mat2Quat(quat, self.data.site_xmat[site_id])
                self.data.qpos[qpos_adr+3:qpos_adr+7] = quat
                self.data.qvel[qvel_adr:qvel_adr+6] = 0

            # --- RENDER ---
            if viewer is not None:
                viewer.sync()
                if self.realtime:
                    time.sleep(self.speed)

    def pick_block(self, block_id=None):
        if block_id is None:
            block_id = self.get_nearest_block()
            mj.mj_forward(self.model, self.data)
            self.attached_block = block_id
        self.attached_block = block_id   

    def put_block(self):
        self.attached_block = None
    def move_gripper_to(self, fingertip_coords):
        result = inverse_kinematics.qpos_from_site_pose(
            self.physics,
            site_name="attachment_site",
            joint_names=["joint"+str(i) for i in range(1,8)],
            target_pos=fingertip_coords,
            max_steps=200
        )

        # ONLY update sim state once
        self.targets = result.qpos[:7]
        #mj.mj_forward(self.model, self.data)
    def get_nearest_block(self, threshold=0.5):
        mj.mj_forward(self.model, self.data)
        site_id = self.model.site("attachment_site").id
        gripper_pos = self.data.site_xpos[site_id]
        nearest = None
        min_dist = float("inf")
        for i in range(self.model.nbody):
            name = self.model.body(i).name
            if not name.startswith("block"):
                continue
            body_id = self.model.body(i).id
            pos = self.data.xpos[body_id]
            dist = np.linalg.norm(pos - gripper_pos)
            if dist < min_dist:
                min_dist = dist
                nearest = body_id
        if min_dist > threshold:
            return None
        return nearest
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