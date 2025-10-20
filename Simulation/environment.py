import pybullet as p
import pybullet_data
import time
import numpy as np

class Env:
    def __init__(self, timestep=1/240.):
        p.connect(p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        p.setTimeStep(timestep)
        p.loadURDF("plane.urdf")

        self.robot_id = p.loadURDF("kuka_iiwa/model.urdf", useFixedBase=True)
        p.resetBasePositionAndOrientation(self.robot_id, [0, 0, 0], [0, 0, 0, 1])
        self.block_ids = []
        self.ee_index = 6  # End effector link
        self.gripper_joints = [7, 8]
        self.arm_joints = list(range(7))
        self.holding_constraint = None
        # Block tracking
        self.block_ids = []
        self.fingertip_coords=[]
    def step(self, steps=240):
        for _ in range(steps):
            p.stepSimulation()
            time.sleep(p.getPhysicsEngineParameters()['fixedTimeStep'])

    def generate_blocks(self, num):
        self.block_ids = []
        for i in range(num):
            block_pos = [0.5, 0.1 * i, 0.05]
            block_id = p.loadURDF("cube_small.urdf", block_pos)
            self.block_ids.append(block_id)

    def pick_block(self, block_id):
        """Attach a block to the end-effector"""
        if self.holding_constraint is not None:
            print("Already holding a block!")
            return
        # Get current end-effector position
        ee_pos, ee_orn = p.getLinkState(self.robot_id, self.ee_index)[:2]
        # Create a fixed constraint to attach the block
        self.holding_constraint = p.createConstraint(
            parentBodyUniqueId=self.robot_id,
            parentLinkIndex=self.ee_index,
            childBodyUniqueId=block_id,
            childLinkIndex=-1,
            jointType=p.JOINT_FIXED,
            jointAxis=[0,0,0],
            parentFramePosition=[0,0,0],
            childFramePosition=[0,0,0]
        )

    def put_block(self):
        """Release the held block"""
        if self.holding_constraint is not None:
            p.removeConstraint(self.holding_constraint)
            self.holding_constraint = None

    def move_gripper_to(self, fingertip_coords, euler=[0, 3.14, 0]):
        fingertip_coords=np.array(fingertip_coords)
        self.fingertip=fingertip_coords.copy()
        fingertip_coords[2]+=0.05
        orn = p.getQuaternionFromEuler(euler)
        joint_angles = p.calculateInverseKinematics(self.robot_id, self.ee_index, fingertip_coords, orn)
        for i in self.arm_joints:
            p.setJointMotorControl2(self.robot_id, i, p.POSITION_CONTROL, joint_angles[i], force=200, maxVelocity=0.9)
        self.step(540)

if __name__=="__main__":
    env=Env()
    env.generate_blocks(4)
    time.sleep(2)
    cube_pos, _ = p.getBasePositionAndOrientation(env.block_ids[0])
    env.move_gripper_to(cube_pos)
    #time.sleep(2)
    env.pick_block(env.block_ids[0])
    up=np.array(cube_pos)
    up[2]+=0.6
    env.move_gripper_to(up)
    time.sleep(10)