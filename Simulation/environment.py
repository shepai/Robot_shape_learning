import pybullet as p
import pybullet_data
import time
import numpy as np

class Env:
    def __init__(self, timestep=1/240.,realtime=False):
        p.connect(p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.realtime=realtime
        self.timestep=timestep
        self.reset()
        
    def step(self, steps=240):
        for _ in range(steps):
            p.stepSimulation()
            if self.realtime:
                time.sleep(p.getPhysicsEngineParameters()['fixedTimeStep'])
    def record(self, filename="simulation.mp4"):
        """Start recording video of the GUI."""
        if not self.recording:
            self.video_id = p.startStateLogging(
                p.STATE_LOGGING_VIDEO_MP4,
                filename
            )
            self.recording = True
            print(f"🎥 Recording started → {filename}")
    def stop_record(self):
        """Stop video recording."""
        if self.recording:
            p.stopStateLogging(self.video_id)
            self.recording = False
            print("📁 Recording stopped and saved.")
    def generate_blocks(self, num): #generate lots of random blocks
        self.block_ids = []
        for i in range(num):
            block_pos = [0.5, 0.1 * i, 0.05]
            block_id = p.loadURDF("cube_small.urdf", block_pos)
            colour = [np.random.random(), np.random.random(), np.random.random(), 1]
            self.block_ids.append(block_id)
            self.positions.append(block_pos)
            self.colours.append(colour)
            self.sizes.append(1)
            p.changeVisualShape(block_id, -1, rgbaColor=colour)
            self.block_file.append("cube_small.urdf")
    def populate(self): #populate like the generate function does, but with all objects
        for i in range(len(self.block_ids)):
            block_pos=self.positions[i]
            if type(self.block_file[i])==type("") and "flat_" in self.block_file[i]:
                collision = p.createCollisionShape(p.GEOM_BOX, halfExtents=self.sizes[i])
                visual = p.createVisualShape(p.GEOM_BOX, halfExtents=self.sizes[i], rgbaColor=self.colours[i])
                block_id = p.createMultiBody(baseMass=1, baseCollisionShapeIndex=collision,
                                baseVisualShapeIndex=visual, basePosition=block_pos)
            else:
                block_id = p.loadURDF(self.block_file[i], block_pos,globalScaling=self.sizes[i])
                p.changeVisualShape(block_id, -1, rgbaColor=self.colours[i])
    def generate_block(self, position,colour,size=1,blockname="cube_small.urdf"): #generate the one random block
        block_id = p.loadURDF(blockname, position, globalScaling=size)
        self.block_ids.append(block_id)
        self.positions.append(position)
        self.block_file.append(blockname)
        self.sizes.append(size)
        self.colours.append(colour)
        p.changeVisualShape(block_id, -1, rgbaColor=colour)
    def pick_block(self, block_id): #pick up the block hovering over
        if self.holding_constraint is not None:
            print("Already holding a block!")
            return

        # Attach directly at the end-effector origin
        self.holding_constraint = p.createConstraint(
            self.robot_id, self.ee_index,
            block_id, -1,
            p.JOINT_FIXED,
            [0, 0, 0],
            [0, 0, 0.03], [0, 0, -0.03]
        )

    def put_block(self):
        """Release the held block"""
        if self.holding_constraint is not None:
            p.removeConstraint(self.holding_constraint)
            self.holding_constraint = None

    def move_gripper_to(self, fingertip_coords, euler=[0, 3.14, 0],vel=0.9): #use kinematics to move towards the robot
        fingertip_coords=np.array(fingertip_coords)
        self.fingertip=fingertip_coords.copy()
        
        orn = p.getQuaternionFromEuler(euler)
        joint_angles = p.calculateInverseKinematics(self.robot_id, self.ee_index, fingertip_coords, orn)
        for i in self.arm_joints:
            p.setJointMotorControl2(self.robot_id, i, p.POSITION_CONTROL, joint_angles[i], force=300, maxVelocity=vel)
        current_pos = np.array(p.getLinkState(self.robot_id, self.ee_index)[0])
        distance = np.linalg.norm(fingertip_coords - current_pos)
        delay_time = distance / vel if vel > 0 else 0
        self.step(max(int(700 * delay_time),1))
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
    def get_observation(self): #return information about the enivornment
        blocks=[]
        colours=[]
        for i in range(len(self.block_ids)):
            cube_pos, _ = p.getBasePositionAndOrientation(self.block_ids[i])
            blocks.append(cube_pos)
            visual_data = p.getVisualShapeData(self.block_ids[i])
            colours.append(visual_data[0][7])
        robot_coords=p.getLinkState(self.robot_id, linkIndex=self.ee_index)[0]
        return {"blocks":blocks,"block_colours":colours,"robot_end_position":robot_coords,"holding_constraint":self.holding_constraint}
    def recreate_from_file(self,env):
        #=load in a file and recreate the objects where they should be
        self.reset()
        self.__dict__.update(env.__dict__)
        self.populate()
    def makeFlat(self,position,colour,length = 0.6,width  = 0.2,height = 0.1,base=1): #make a flat block
        halfExtents = [length / 2, width / 2, height / 2]
        collision = p.createCollisionShape(p.GEOM_BOX, halfExtents=halfExtents)
        visual = p.createVisualShape(p.GEOM_BOX, halfExtents=halfExtents, rgbaColor=colour)
        block_id = p.createMultiBody(baseMass=base, baseCollisionShapeIndex=collision,
                        baseVisualShapeIndex=visual, basePosition=position)
        self.block_ids.append(block_id)
        self.positions.append(position)
        self.block_file.append("flat_")
        self.sizes.append(halfExtents)
        self.colours.append(colour)
        p.changeVisualShape(block_id, -1, rgbaColor=colour)

    def reset(self): #reset the simulation and arm
        p.resetSimulation()
        p.setGravity(0, 0, -9.81)
        p.setTimeStep(self.timestep)
        p.loadURDF("plane.urdf")
        self.block_file=[]
        self.positions=[]
        self.colours=[]
        self.sizes=[]
        self.block_ids = []
        self.robot_id = p.loadURDF("kuka_iiwa/model.urdf", useFixedBase=True)
        p.resetBasePositionAndOrientation(self.robot_id, [0, 0, 0], [0, 0, 0, 1])
        self.ee_index = 6  # End effector link
        self.gripper_joints = [7, 8]
        self.arm_joints = list(range(7))
        self.holding_constraint = None
        # Block tracking
        
        self.fingertip_coords=[]
        self.recording = False
        self.video_id = None
        self.timestep = self.timestep
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)              # remove side panels
        p.configureDebugVisualizer(p.COV_ENABLE_RGB_BUFFER_PREVIEW, 0)
        p.configureDebugVisualizer(p.COV_ENABLE_DEPTH_BUFFER_PREVIEW, 0)
        p.configureDebugVisualizer(p.COV_ENABLE_SEGMENTATION_MARK_PREVIEW, 0)
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)        # keep rendering on
        p.resetDebugVisualizerCamera(
            cameraDistance=1.5,        # smaller = closer zoom
            cameraYaw=45,              # rotate left/right
            cameraPitch=-30,           # tilt up/down
            cameraTargetPosition=[0,0,0.5]  # where the camera looks
        )
        self.move_Step_amount=0.2
        self.realtime=self.realtime
    def close(self):
        p.disconnect()
if __name__=="__main__":
    env=Env()
    env.generate_blocks(4)
    env.record("/its/home/drs25/Documents/GitHub/Robot_shape_learning/Assets/Videos/video_example.mp4")
    time.sleep(2)
    cube_pos, _ = p.getBasePositionAndOrientation(env.block_ids[0])
    cube_pos=list(cube_pos)
    cube_pos[2]+=0.08
    env.move_gripper_to(cube_pos)
    #time.sleep(2)
    env.pick_block(env.block_ids[0])
    up=np.array(cube_pos)
    up[2]+=0.6
    env.move_gripper_to(up)
    cube_pos, _ = p.getBasePositionAndOrientation(env.block_ids[1])
    cube_pos=np.array(cube_pos)
    cube_pos[2]+=0.18
    env.move_gripper_to(cube_pos)
    env.put_block()
    env.move_gripper_to(up)
    env.stop_record()
    functions=[env.forward,env.backward,env.move_up,env.move_down,env.left,env.right]
    for i in range(len(functions)):
        for j in range(10):
            functions[i]()
    env.reset()
    env.populate()
    env.step(10000)
    time.sleep(10)
