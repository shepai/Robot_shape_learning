import pybullet as p
import pybullet_data
import time

# --- Setup ---
p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.81)

p.loadURDF("plane.urdf")

# Load the KUKA arm with gripper
robot_ids = p.loadSDF("kuka_iiwa/kuka_with_gripper.sdf")
robot_id = robot_ids[0]
p.resetBasePositionAndOrientation(robot_id, [0, 0, 0], [0, 0, 0, 1])

# Spawn a cube to pick
cube_id = p.loadURDF("cube_small.urdf", [0.7, 0, 0.05])

# --- Discover joints ---
print("Number of joints:", p.getNumJoints(robot_id))
for i in range(p.getNumJoints(robot_id)):
    info = p.getJointInfo(robot_id, i)
    print(i, info[1].decode("utf-8"))

# Typical joint indices:
#  0–6 : arm joints
#  7–8 : gripper finger joints

ee_index = 6  # End-effector link

# --- Move above the cube ---
target_pos = [0.7, 0, 0.2]
target_orn = p.getQuaternionFromEuler([0, 3.14, 0])  # Palm down
joint_angles = p.calculateInverseKinematics(robot_id, ee_index, target_pos, target_orn)

for i in range(7):
    p.setJointMotorControl2(robot_id, i, p.POSITION_CONTROL, joint_angles[i], force=200)

for _ in range(240):
    p.stepSimulation()
    time.sleep(1/240)

# --- Lower onto the cube ---
target_pos[2] = 0.05
joint_angles = p.calculateInverseKinematics(robot_id, ee_index, target_pos, target_orn)
for i in range(7):
    p.setJointMotorControl2(robot_id, i, p.POSITION_CONTROL, joint_angles[i], force=200)

for _ in range(240):
    p.stepSimulation()
    time.sleep(1/240)

# --- Close the gripper ---
finger_left = 7
finger_right = 8
p.setJointMotorControl2(robot_id, finger_left, p.POSITION_CONTROL, 0.02, force=50)
p.setJointMotorControl2(robot_id, finger_right, p.POSITION_CONTROL, 0.02, force=50)

for _ in range(240):
    p.stepSimulation()
    time.sleep(1/240)

# --- Lift the cube ---
target_pos[2] = 0.25
joint_angles = p.calculateInverseKinematics(robot_id, ee_index, target_pos, target_orn)
for i in range(7):
    p.setJointMotorControl2(robot_id, i, p.POSITION_CONTROL, joint_angles[i], force=200)

for _ in range(480):
    p.stepSimulation()
    time.sleep(1/240)

input("Press Enter to exit...")
p.disconnect()
