"""
The task code has a number of challenges where we know what a correct solution is
"""

from copy import deepcopy 
import time 
import pickle
import numpy as np

class task:
    def __init__(self):
        self.attempts=0
        self.best_time=0
        self.timer=0
        self.trial=0
        self.fitnesshistory=[]
        self.currentfit=[]
    def get_correctness(self,observations):
        """
        Takes in observation
        """
        return 0,{}
    def reset(self):
        self.fitnesshistory.append(deepcopy(self.currentfit))
        self.trial+=1
        self.timer=0
        self.currentfit=[]
    def save(self,filepath):
        if ".pickle" not in filepath: filepath+=".pickle"
        with open(filepath, 'wb') as handle:
            pickle.dump(self.fitnesshistory, handle, protocol=pickle.HIGHEST_PROTOCOL)
    def possible_solutions(self):
        pass 
    def generate(self,p):
        pass
    def save_details(self):
        #save the objects, start positions, trial number, possible outcome 
        #csv and link file to environment 
        pass
class task1(task):
    """
    Make a tower with all the blocks (no particular order)
    """
    def generate(self,env):
        """
        Generate blocks in random places
        """
        amount=np.random.randint(2,7)
        blocks=np.zeros((amount,3))
        min_dist = 0.1
        for i in range(amount):
            not_allowed=True 
            t1=time.time()
            t2=time.time()
            while not_allowed and t2-t1<5:
                t2=time.time()
                position = np.array([np.random.uniform(low=0.4, high=0.6), np.random.uniform(low=0.0, high=0.6), 0.05])
                # Check distances from all existing blocks
                if len(blocks) == 0:
                    blocks.append(position)
                    break
                distances = [np.linalg.norm(position[:2] - b[:2]) for b in blocks]
                if all(d > min_dist for d in distances):
                    not_allowed=False

            blocks[i]=position
            env.generate_block(position,[np.random.random(), np.random.random(), np.random.random(), 1])
    def solve(self,env,p):
        in_order=[]
        distances=[]
        for i in range(len(env.block_ids)): #sort the ids in order so we take the furthest ones first
            target_pos, _ = p.getBasePositionAndOrientation(env.block_ids[i])
            #insertion sort 
            distance=np.linalg.norm(np.array([0,0,0])-target_pos)
            if len(distances)==0:
                distances.append(distance)
                in_order.append(i)
            else:
                change_made=True
                j=0
                while j<(len(distances)) and change_made: #insertion
                    if distance>distances[j]:
                        distances.insert(j,distance)
                        in_order.insert(j,i)
                        change_made=not change_made
                    j+=1
                if j>=len(distances): 
                    distances.append(distance)
                    in_order.append(i)

        target_pos=[np.random.uniform(low=0.4, high=0.6), np.random.uniform(low=-0.3, high=0.0), 0.10]
        for i in range(len(env.block_ids)):
            cube_pos, _ = p.getBasePositionAndOrientation(env.block_ids[in_order[i]]) #find id
            cube_pos=list(cube_pos)
            cube_pos[2]+=0.08
            env.move_gripper_to(cube_pos) #move to just above it
            env.pick_block(env.block_ids[in_order[i]]) #pick up
            cube_pos[2]+=0.38
            env.move_gripper_to(cube_pos) #move up to avoid hitting into things
            env.step(10)
            cube_pos[0:2]=deepcopy(target_pos[0:2])
            env.move_gripper_to(cube_pos) #move up to avoid hitting into things
            env.step(10)
            env.move_gripper_to(target_pos) #move to the target
            env.step(15) #small delay
            env.put_block() #release
            env.move_gripper_to(cube_pos)
            target_pos[2]+=0.05 #move upwards
            
    def get_correctness(self,obs):
        #find the maximum z of the blocks and work out how many are touching
        blocks=np.array(obs['blocks'])
        return 1-np.average([np.std(blocks[:,1]),np.std(blocks[:,0])])

class task2(task):
    """
    Make a tower with all the blocks (no particular order)
    """
    def generate(self,env):
        """
        Generate blocks in random places
        """
        colours=[[255,0,0,1],[0,255,0,1],[0,0,255,1]]
        amount=np.random.randint(5,15)
        blocks=np.zeros((amount,3))
        min_dist = 0.05
        for i in range(amount):
            not_allowed=True 
            t1=time.time()
            t2=time.time()
            while not_allowed and t2-t1<5:
                t2=time.time()
                position = np.array([np.random.uniform(low=0.4, high=0.6), np.random.uniform(low=0.0, high=0.6), 0.05])
                # Check distances from all existing blocks
                if len(blocks) == 0:
                    blocks.append(position)
                    break
                distances = [np.linalg.norm(position[:2] - b[:2]) for b in blocks]
                if all(d > min_dist for d in distances):
                    not_allowed=False

            blocks[i]=position
            env.generate_block(position,colours[np.random.randint(0,3)])
    def solve(self,env,p):
        point=np.array([np.random.uniform(low=0.4, high=0.6), np.random.uniform(low=-0.6, high=0.0), 0.10])
        targets=[]
        heights=[0,0,0]
        for i in range(3):
            targets.append(deepcopy(point))
            targets[i][0]+= 0.3 * i
        print(targets)
        targetidx=-1
        for i in range(len(env.block_ids)):
            visual_data = p.getVisualShapeData(env.block_ids[i])
            colour=visual_data[0][7]
            if list(colour)==[255,0,0,1]:
                targetidx=0
            elif list(colour)==[0,255,0,1]:
                targetidx=1
            elif list(colour)==[0,0,255,1]:
                targetidx=2
            print(targetidx,"\n\n")
            target_pos=deepcopy(targets[targetidx])
            target_pos[2]+=heights[targetidx]
            print(target_pos)
            cube_pos, _ = p.getBasePositionAndOrientation(env.block_ids[i]) #find id
            cube_pos=list(cube_pos)
            cube_pos[2]+=0.08
            env.move_gripper_to(cube_pos) #move to just above it
            env.pick_block(env.block_ids[i]) #pick up
            cube_pos[0:2]=target_pos[0:2]
            cube_pos[2]+=0.38
            env.move_gripper_to(cube_pos) #move up to avoid hitting into things
            env.step(10)
            env.move_gripper_to(target_pos) #move to the target
            env.step(15) #small delay
            env.put_block() #release
            env.move_gripper_to(cube_pos)
            #target_pos[2]+=0.05
            heights[targetidx]+=0.5

    def get_correctness(self,obs):
        #find the average groupings
        pass
    
if __name__=="__main__":
    from environment import *
    env=Env(realtime=1)
    task=task2()
    task.generate(env)
    #env.record("/its/home/drs25/Documents/GitHub/Robot_shape_learning/Assets/Videos/task2.mp4")
    print("Correctness value:",task.get_correctness(env.get_observation()))
    task.solve(env,p)
    print("Correctness value:",task.get_correctness(env.get_observation()))
    #env.stop_record() 
