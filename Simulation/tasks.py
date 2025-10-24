"""
The task code has a number of challenges where we know what a correct solution is. Each task has its own destinct reward function to determine how well the 
task was performed. This can also be used to generate a dataset of itmes and task. 

Task 1: arrange into a tower <<

Task 2: sort the colours into three seperate towers <

Task 3: Sort the shades into intensity order <

Task 4: Sort into actual size order <

Task 5: place the ball on top of the box

Task 6: Sort the missing pieces

Task 7: Put the sizes in the right piles

Task 8: Sort size in one axis and colour in the other

Task 9: Look for the lowest colour and remove it from the pile

Task 10: Place the four blocks given into a square 

Task 11: avoid flat surface to pick up objects underneth it and place them on top

Task 12: Push on opposite side of bloc kresting on cylender, and it should be up in the air
"""

from copy import deepcopy 
import time 
import pickle
import numpy as np
class demo:
    def __init__(self,env,task):
        self.env=env 
        self.task=task
class task:
    def __init__(self):
        self.attempts=0
        self.best_time=0
        self.timer=0
        self.trial=0
        self.fitnesshistory=[]
        self.currentfit=[]
        self.loaded=False
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
    def save_details(self,filename,env):
        #save the objects, start positions, trial number, possible outcome 
        #csv and link file to environment 
        to_save=demo(env,self)
        with open(filename, 'wb') as f:
            pickle.dump(to_save, f)
    def load_details(self,filename,env):
        if ".pkl" not in filename: filename+=".pkl"
        with open(filename, 'rb') as f:
            to_save = pickle.load(f)
        self.__dict__.update(to_save.task.__dict__)
        #update environment
        env.recreate_from_file(to_save.env)
        self.loaded=True


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
            env.step(10)
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
    Make a tower with all the blocks (colour particular order)
    """
    def generate(self,env):
        """
        Generate blocks in random places
        """
        colours=[[1,0,0,1],[0,1,0,1],[0,0,1,1]]
        amount=np.random.randint(5,15)
        blocks=np.zeros((amount,3))
        min_dist = 0.08
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
        point=np.array([np.random.uniform(low=0.3, high=0.4), np.random.uniform(low=-0.6, high=0.0), 0.10])
        targets=[]
        heights=[0,0,0]
        for i in range(3):
            targets.append(deepcopy(point))
            targets[i][0]+= 0.3 * i
        targetidx=-1
        for i in range(len(env.block_ids)):
            visual_data = p.getVisualShapeData(env.block_ids[i])
            colour=visual_data[0][7]
            if list(colour)==[1,0,0,1]:
                targetidx=0
            elif list(colour)==[0,1,0,1]:
                targetidx=1
            elif list(colour)==[0,0,1,1]:
                targetidx=2
            target_pos=deepcopy(targets[targetidx])
            target_pos[2]+=heights[targetidx]
            cube_pos, _ = p.getBasePositionAndOrientation(env.block_ids[i]) #find id
            cube_pos=list(cube_pos)
            cube_pos[2]+=0.08
            env.move_gripper_to(cube_pos) #move to just above it
            env.step(10)
            env.pick_block(env.block_ids[i]) #pick up
            cube_pos[2]+=0.38
            env.move_gripper_to(cube_pos) #move up to avoid hitting into things
            env.step(10)
            cube_pos[0:2]=target_pos[0:2]
            env.move_gripper_to(cube_pos) #move up to avoid hitting into things
            env.step(10)
            env.move_gripper_to(target_pos) #move to the target
            env.step(15) #small delay
            env.put_block() #release
            env.move_gripper_to(cube_pos)
            #target_pos[2]+=0.05
            heights[targetidx]+=0.05

    def get_correctness(self,obs):
        #find the average groupings
        for i in range(len(env.block_ids)):
            visual_data = p.getVisualShapeData(env.block_ids[i])
            colour=visual_data[0][7]
        #TODO
class task3(task):
    """
    Make a line with all the colours 
    """
    def generate(self,env):
        colour=[np.random.randint(0,254)/255,np.random.randint(0,254)/255,np.random.randint(0,254)/255,1]
        shader=np.random.randint(0,3)
        amount=np.random.randint(5,10)
        blocks=np.zeros((amount,3))
        min_dist = 0.08
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
            colour[shader]-=(5*i)/255
            if colour[shader]<0: colour[shader]=1-(5*i)/255
            env.generate_block(position,deepcopy(colour))
    def solve(self,env,p):
        sorted_ids=[] 
        shades=[]
        for i in range(len(env.block_ids)):
            visual_data = p.getVisualShapeData(env.block_ids[i])
            colour=visual_data[0][7]
            grey_intensity=0.299*colour[0]+0.587*colour[1]+0.114*colour[2]
            if len(shades)==0:
                shades.append(grey_intensity)
                sorted_ids.append(i)
            else:
                change_made=True
                j=0
                while j<(len(shades)) and change_made: #insertion
                    if grey_intensity>shades[j]:
                        shades.insert(j,grey_intensity)
                        sorted_ids.insert(j,i)
                        change_made=not change_made
                    j+=1
                if j>=len(shades): 
                    shades.append(grey_intensity)
                    sorted_ids.append(i)
        target_location=np.array([np.random.uniform(low=0.5, high=0.6), np.random.uniform(low=-0.1, high=0.0), 0.10])

        for i in range(len(env.block_ids)): 
            place_in_queue=sorted_ids.index(i)
            temp_loc=deepcopy(target_location)
            temp_loc[1]=temp_loc[1]-(place_in_queue*0.15)
            cube_pos, _ = p.getBasePositionAndOrientation(env.block_ids[i]) #find id
            cube_pos=list(cube_pos)
            cube_pos[2]+=0.08
            env.move_gripper_to(cube_pos) #move to just above it
            env.step(10)
            env.pick_block(env.block_ids[i]) #pick up
            cube_pos[2]+=0.38
            env.move_gripper_to(cube_pos) #move up to avoid hitting into things
            env.step(10)
            cube_pos[0:2]=temp_loc[0:2]
            env.move_gripper_to(cube_pos) #move up to avoid hitting into things
            env.step(10)
            env.move_gripper_to(temp_loc) #move to the target
            env.step(15) #small delay
            env.put_block() #release
            env.move_gripper_to(cube_pos)
        

    def get_correctness(self,obs):
        #should be in correct order
        #TODO
        pass

class task4(task):
    def generate(self, env):
        amount=np.random.randint(5,10)
        sizes=np.random.uniform(0.5,2,(amount))
        blocks=np.zeros((amount,3))
        colour=[np.random.randint(0,254)/255,np.random.randint(0,254)/255,np.random.randint(0,254)/255,1]
        min_dist = 0.12
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
            env.generate_block(position,deepcopy(colour),sizes[i])
    def solve(self,env,p):
        sorted_ids=[] 
        shades=[]
        for i in range(len(env.block_ids)):
            aabb_min, aabb_max = p.getAABB(env.block_ids[i])
            size = [aabb_max[i] - aabb_min[i] for i in range(3)]
            if len(shades)==0:
                shades.append(size)
                sorted_ids.append(i)
            else:
                change_made=True
                j=0
                while j<(len(shades)) and change_made: #insertion
                    if size>shades[j]:
                        shades.insert(j,size)
                        sorted_ids.insert(j,i)
                        change_made=not change_made
                    j+=1
                if j>=len(shades): 
                    shades.append(size)
                    sorted_ids.append(i)
        target_location=np.array([np.random.uniform(low=0.5, high=0.6), np.random.uniform(low=-0.1, high=0.0), 0.10])

        for i in range(len(env.block_ids)): 
            place_in_queue=sorted_ids.index(i)
            temp_loc=deepcopy(target_location)
            temp_loc[1]=temp_loc[1]-(place_in_queue*0.15)
            cube_pos, _ = p.getBasePositionAndOrientation(env.block_ids[i]) #find id
            cube_pos=list(cube_pos)
            cube_pos[2]+=0.08
            env.move_gripper_to(cube_pos) #move to just above it
            env.step(10)
            env.pick_block(env.block_ids[i]) #pick up
            cube_pos[2]+=0.38
            env.move_gripper_to(cube_pos) #move up to avoid hitting into things
            env.step(10)
            cube_pos[0:2]=temp_loc[0:2]
            env.move_gripper_to(cube_pos) #move up to avoid hitting into things
            env.step(10)
            env.move_gripper_to(temp_loc) #move to the target
            env.step(15) #small delay
            env.put_block() #release
            env.move_gripper_to(cube_pos)
    def get_correctness(self,obs):
        #should be in correct order
        #TODO  
        pass

class task5(task):
    def generate(self, env):
        colour=[np.random.randint(0,254)/255,np.random.randint(0,254)/255,np.random.randint(0,254)/255,1]
        env.generate_block([np.random.random(),np.random.random(),np.random.random()],colour,1,"sphere_small.urdf")
        env.makeFlat([-0.5,-0.5,0.2],colour,length = 0.6,width  = 0.6,height = 0.1,base=0)
    def solve(self,env,p):
        ball=None 
        block=None
        for i in range(len(env.block_ids)):
            if type(env.block_file[i])==type("") and "flat_" in env.block_file[i]:
                block=env.block_ids[i]
            else: 
                ball=env.block_ids[i]
        cube_pos, _ = p.getBasePositionAndOrientation(ball) #find id
        temp_loc, _ = p.getBasePositionAndOrientation(block) #find id
        cube_pos=list(cube_pos)
        cube_pos[2]+=0.08
        env.move_gripper_to(cube_pos) #move to just above it
        env.step(10)
        env.pick_block(env.block_ids[i]) #pick up
        cube_pos[2]+=0.38
        env.move_gripper_to(cube_pos) #move up to avoid hitting into things
        env.step(10)
        cube_pos[0:2]=temp_loc[0:2]
        env.move_gripper_to(cube_pos) #move up to avoid hitting into things
        env.step(10)
        env.move_gripper_to(temp_loc) #move to the target
        env.step(15) #small delay
        env.put_block() #release
        env.move_gripper_to(cube_pos)
    def get_correctness(self,obs):
        #should be in correct order
        #TODO  
        pass
if __name__=="__main__":
    from environment import *
    env=Env(realtime=0)
    task=task5()
    task.generate(env)
    #env.record("/its/home/drs25/Documents/GitHub/Robot_shape_learning/Assets/Videos/task5.mp4")
    print("Correctness value:",task.get_correctness(env.get_observation()))
    task.solve(env,p)
    print("Correctness value:",task.get_correctness(env.get_observation()))
    #env.stop_record() 
    task.save_details("/its/home/drs25/Documents/GitHub/Robot_shape_learning/Assets/Data/example.pkl",env)
    #test save
    env.close()
    del env 
    del task 
    env=Env(realtime=0)
    task=task5()
    task.load_details("/its/home/drs25/Documents/GitHub/Robot_shape_learning/Assets/Data/example.pkl",env)
    task.solve(env,p)