"""
The task code has a number of challenges where we know what a correct solution is. Each task has its own destinct reward function to determine how well the 
task was performed. This can also be used to generate a dataset of itmes and task. 

Task 1: arrange into a tower <<

Task 2: sort the colours into three seperate towers <<

Task 3: Sort the shades into intensity order <<

Task 4: Sort into actual size order <<

Task 5: place the ball on top of the box <<

Task 6: Sort the missing pieces <<

Task 7: Put the sizes in the right piles <<

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
        blocks=[]
        min_dist = 0.1
        for i in range(amount):
            not_allowed=True 
            t1=time.time()
            t2=time.time()
            while not_allowed and t2-t1<2:
                t2=time.time()
                position = np.array([np.random.uniform(low=0.4, high=0.6), np.random.uniform(low=0.0, high=0.6), 0.05])
                # Check distances from all existing blocks
                if len(blocks) == 0:
                    blocks.append(position)
                    break
                distances = [np.linalg.norm(position[:2] - b[:2]) for b in np.array(blocks)]
                if all(d > min_dist for d in distances):
                    not_allowed=False
            if t2-t1<2:
                blocks.append(position)
                env.generate_block(position,[np.random.random(), np.random.random(), np.random.random(), 1])
        env.step(100)
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

        target_pos=[np.random.uniform(low=0.6, high=0.8), np.random.uniform(low=-0.3, high=-0.1), 0.10]
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
        blocks=[]
        min_dist = 0.15
        for i in range(amount):
            not_allowed=True 
            t1=time.time()
            t2=time.time()
            while not_allowed and t2-t1<2:
                t2=time.time()
                position = np.array([np.random.uniform(low=0.5, high=0.6), np.random.uniform(low=0.0, high=0.6), 0.05])
                # Check distances from all existing blocks
                if len(blocks) == 0:
                    blocks.append(position)
                    break
                distances = [np.linalg.norm(position[:2] - b[:2]) for b in np.array(blocks)]
                if all(d > min_dist for d in distances):
                    not_allowed=False
            if t2-t1<2:
                blocks.append(position)
                env.generate_block(position,colours[np.random.randint(0,3)])
            else:
                break
        env.step(100)
    def solve(self,env,p):
        point=np.array([np.random.uniform(low=0.4, high=0.5), np.random.uniform(low=-0.6, high=0.0), 0.10])
        targets=[]
        heights=[0,0,0]
        for i in range(3):
            targets.append(deepcopy(point))
            targets[i][0]+= 0.2 * i
        targetidx=-1
        for i in range(len(env.block_ids)):
            #print("Correctness value:",self.get_correctness(env.get_observation()))
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
        from sklearn.cluster import KMeans
        from sklearn.metrics import adjusted_rand_score
        #find the average groupings
        positions=obs['blocks']
        colours=obs['block_colours']
 
        positions = np.array(positions)
        colours = np.array(colours, dtype=float)
        num_colours,colour_labels=np.unique(colours, axis=0, return_inverse=True)

        n_clusters=len(num_colours)
        if colours.max() > 1:  # normalize colour range
            colours /= 255.0

        kmeans = KMeans(n_clusters=n_clusters, n_init='auto').fit(positions)
        spatial_labels = kmeans.labels_
        centers = kmeans.cluster_centers_
        dists = np.linalg.norm(positions - centers[spatial_labels], axis=1)
        spatial_score = 1 - np.clip(np.mean(dists) / (np.max(dists) + 1e-9), 0, 1)
        alignment_score = adjusted_rand_score(colour_labels, spatial_labels)
        final_score = 0.6 * spatial_score + 0.4 * alignment_score
        return float(final_score)
class task3(task):
    """
    Make a line with all the colours 
    """
    def generate(self,env):
        colour=[np.random.randint(0,254)/255,np.random.randint(0,254)/255,np.random.randint(0,254)/255,1]
        shader=np.random.randint(0,3)
        amount=np.random.randint(5,10)
        blocks=[]
        min_dist = 0.08
        for i in range(amount):
            not_allowed=True 
            t1=time.time()
            t2=time.time()
            while not_allowed and t2-t1<2:
                t2=time.time()
                position = np.array([np.random.uniform(low=0.4, high=0.6), np.random.uniform(low=0.0, high=0.6), 0.05])
                # Check distances from all existing blocks
                if len(blocks) == 0:
                    blocks.append(position)
                    break
                distances = [np.linalg.norm(position[:2] - b[:2]) for b in np.array(blocks)]
                if all(d > min_dist for d in distances):
                    not_allowed=False
            
            if t2-t1<2:
                env.generate_block(position,deepcopy(colour))
                blocks.append(position)
                colour[shader]-=(5*i)/255
                if colour[shader]<0: colour[shader]=1-(5*i)/255
            else:
                break
        env.step(100)
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
        from scipy.spatial.distance import cdist
        from scipy.stats import kendalltau

        coords = np.array(obs['blocks'])
        colours=np.array(obs['block_colours'])
        intensity = 0.299 * colours[:, 0] + 0.587 * colours[:, 1] + 0.114 * colours[:, 2]
        sizes = intensity

        n = len(coords)
        remaining = list(range(n))
        order = [remaining.pop(0)]

        while remaining:
            last = order[-1]
            rem_arr = np.array(remaining, dtype=int)
            dists = cdist(coords[[last]], coords[rem_arr])[0]
            nearest_idx = np.argmin(dists)
            nearest = remaining[nearest_idx]
            order.append(nearest)
            remaining.remove(nearest)

        spatial_sizes = sizes[order]
        tau, _ = kendalltau(np.arange(n), spatial_sizes)
        score = abs(tau)  # allow either increasing or decreasing

        return score

class task4(task):
    """
    Place in the size of the block order
    """
    def generate(self, env):
        amount=np.random.randint(5,10)
        sizes=np.random.uniform(0.5,2,(amount))
        blocks=[]
        colour=[np.random.randint(0,254)/255,np.random.randint(0,254)/255,np.random.randint(0,254)/255,1]
        min_dist = 0.15
        for i in range(amount):
            not_allowed=True 
            t1=time.time()
            t2=time.time()
            while not_allowed and t2-t1<2:
                t2=time.time()
                position = np.array([np.random.uniform(low=0.4, high=0.6), np.random.uniform(low=0.0, high=0.6), 0.05])
                # Check distances from all existing blocks
                if len(blocks) == 0:
                    blocks.append(position)
                    break
                distances = [np.linalg.norm(position[:2] - b[:2]) for b in blocks]
                if all(d > min_dist for d in np.array(distances)):
                    not_allowed=False
                    print("not allowed")
            if t2-t1<2:
                blocks.append(position)
                env.generate_block(position,deepcopy(colour),sizes[i])
            else:
                break
        env.step(100)
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
    def get_correctness(self, obs):
        from scipy.spatial.distance import cdist
        from scipy.stats import kendalltau

        coords = np.array(obs['blocks'])
        sizes = np.array(obs['sizes'])

        n = len(coords)
        remaining = list(range(n))
        order = [remaining.pop(0)]

        while remaining:
            last = order[-1]
            rem_arr = np.array(remaining, dtype=int)
            dists = cdist(coords[[last]], coords[rem_arr])[0]
            nearest_idx = np.argmin(dists)
            nearest = remaining[nearest_idx]
            order.append(nearest)
            remaining.remove(nearest)

        spatial_sizes = sizes[order]
        tau, _ = kendalltau(np.arange(n), spatial_sizes)
        score = abs(tau)  # allow either increasing or decreasing

        return score
        
        
class task5(task):
    """
    Place the ball on the board
    """
    def generate(self, env):
        colour=[np.random.randint(0,254)/255,np.random.randint(0,254)/255,np.random.randint(0,254)/255,1]
        x_bounds=min(0.4,max(0.7,np.random.random()))
        y_bounds=min(0.4,max(0.7,np.random.random()))
        env.generate_block([x_bounds,y_bounds,0.1],colour,1,"sphere_small.urdf")
        env.makeFlat([0.5,-0.5,0.2],colour,length = 0.6,width  = 0.6,height = 0.1,base=0)
        env.step(100)
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
        temp_loc=list(temp_loc)
        temp_loc[2]+=0.15
        cube_pos=list(cube_pos)
        cube_pos[2]+=0.08
        env.move_gripper_to(cube_pos) #move to just above it
        env.step(10)
        env.pick_block(ball) #pick up
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
        ball=None 
        block=None
        block_pos=obs["blocks"]
        names=obs["block_name"]
        for i in range(len(block_pos)):
            if type(names[i])==type("") and "flat_" in names[i]:
                block=i
            else: 
                ball=i
        if block_pos[ball][2]>block_pos[block][2]: #this is good
            #check is touching
            if len(obs['contacts'][0])>0:
                return 1
            #return inverse distance to platform as a guide
            p1=np.array(block_pos[ball])
            p2=np.array(block_pos[block])
            return 1- np.linalg.norm(p2 - p1)
        return 0
class task6(task):
    """
    Fill in the missing pieces
    """
    def generate(self, env):
        grid_start=np.array([0.3,-0.3,0.1])+np.random.normal(0,0.01,(3,))
        remove=[np.random.randint(0,9) for i in range(np.random.randint(1,3))]

        c=0
        colours=[[1,0,0,1],[0,1,0,1],[0,0,1,1]]
        ind1=np.random.randint(0,3)
        ind2=np.random.randint(0,3)
        self.targets=[]
        self.blocks=[]
        self.id=[]
        for i in range(4):
            for j in range(4):
                shift=grid_start.copy() 
                shift[0]+=0.1*i
                shift[1]-=0.1*j
                colour = colours[ind1] if c%2==0 else colours[ind2]
                if c not in remove:
                    env.generate_block(shift,deepcopy(colour))
                else: 
                    coords=[np.random.uniform(low=0.4, high=0.6), np.random.uniform(low=0.0, high=0.6), 0.05]
                    env.generate_block(coords,deepcopy(colour))
                    self.targets.append(shift)
                    self.blocks.append(coords)
                    self.id.append(c)
                c+=1 
                #grid[i][j]=shift.copy()
        
        
        #make lines of the same colour with missing slots, and then the missing pieces in another pile

    def solve(self,env,p):
        def find_nearest(d, value, tol=0.01):
            # find key in dict d closest to value within tolerance tol
            keys = np.array(list(d.keys()))
            diff = np.abs(keys - value)
            if len(diff) == 0 or np.min(diff) > tol:
                return None
            return keys[np.argmin(diff)]
        #calculate the mising spots and which colour it is
        objects=[]
        count_coords_x={}
        count_coords_y={}
        id_tracker_x={}
        id_tracker_y={}
        for i in range(len(env.block_ids)): #loop through and work out what is in the grid and what is not
            cube_pos, _ = p.getBasePositionAndOrientation(env.block_ids[i]) #find id
            #cube_pos=[round(cube_pos[0],4),round(cube_pos[1],4),round(cube_pos[2],4)]
            objects.append(cube_pos)
            coord = (round(cube_pos[0],4), round(cube_pos[1],4))
            count_coords_x[coord[0]] = count_coords_x.get(coord[0], 0) + 1
            count_coords_y[coord[1]] = count_coords_y.get(coord[1], 0) + 1
            id_tracker_x[coord[0]] = id_tracker_x.get(coord[0], [])
            id_tracker_x[coord[0]].append(i)
            id_tracker_y[coord[1]] = id_tracker_y.get(coord[1], [])
            id_tracker_y[coord[1]].append(i)
        to_pickup=[]
        for key in count_coords_x: #find the target ones
            if count_coords_x[key]<=1:
                for key2 in count_coords_y:
                    if count_coords_y[key2]<=1:
                        to_pickup.append(id_tracker_x[key][0])
        #reform the grid without the blocks
        original_indices = np.arange(len(objects))
        ids=np.unique(np.array(to_pickup))
        size=int(np.sqrt(len(objects)))
        mask = np.ones(len(objects), dtype=bool)
        mask[ids] = False
        objects = np.array(objects)[mask]
        original_indices = original_indices[mask]
        xmin, xmax = np.min(objects[:,0]), np.max(objects[:,0])
        ymin, ymax = np.min(objects[:,1]), np.max(objects[:,1])
        targets=[]
        distx = round((xmax - xmin) / (size - 1), 4)
        disty = round((ymax - ymin) / (size - 1), 4)
        dist=max(distx,disty)
        c=0
        grid=np.zeros((size,size))-1
        for x in range(size): #loop through potential grid
            for y in range(size):
                current_x=xmin+dist*x #calculate estimated positions
                current_y=ymax-dist*y
                x_key = find_nearest(id_tracker_x, round(current_x,4)) #look to see if it is in the grid
                y_key = find_nearest(id_tracker_y, round(current_y,4))
                relatedx=id_tracker_x.get(x_key)
                relatedy=id_tracker_y.get(y_key)
                if type(relatedx)!=type(None) and type(relatedy)!=type(None):
                    overlap = np.intersect1d(relatedx, relatedy)
                    if len(overlap) > 0:
                        # cell occupied
                        visual_data = p.getVisualShapeData(env.block_ids[c])
                        colour=visual_data[0][7][:-1]
                        grid[x][y]=np.argmax(colour)
                    else:
                        # cell empty → target
                        targets.append([current_x, current_y, 0.1, c, x, y])

                c+=1
        y, x = np.where(grid == -1) #get the targets
        #fill in the missing colours
        #check the index and colour its meant to be
        new_targets=[]
        ids=[]
        for yi, xi in zip(y, x):
            col = grid[:, xi]
            vals, counts = np.unique(col[col != -1], return_counts=True)
            correct_value = vals[np.argmax(counts)]  # mode of the column
            
            grid[yi, xi] = correct_value
            colour=np.array([0,0,0,1])
            colour[int(correct_value)]=1
            for i in range(len(targets)):
                if targets[i][4]==yi and targets[i][5]==xi: #put in corret order
                    new_targets.append(targets[i].copy())
                    id_=targets[i][3]
                    cube_pos, _ = p.getBasePositionAndOrientation(env.block_ids[id_]) #find id
                    ids.append(cube_pos)
        for i in range(len(targets)): #place the target from the og position to the target
            #print("Correctness value:",self.get_correctness(env.get_observation()))
            temp_loc=list(new_targets[i][0:3])
            cube_pos=list(ids[i])
            cube_pos[2]+=0.08
            env.move_gripper_to(cube_pos) #move to just above it
            env.step(10)
            env.pick_block(env.block_ids[new_targets[i][3]]) #pick up
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
        #place correct colour in spaces 
    def get_correctness(self,obs):
        #should be in correct order
        #distance from target one plus distance to target 2 divided by 2 and normalised (over total distances)
        dist1=0
        for i in range(len(self.targets)):
            a=np.array(self.targets[i][0:2])
            b=np.array(self.blocks[i][0:2])
            total=np.linalg.norm(a-b)
            current=np.array(obs['blocks'][self.id[i]][0:2])
            dist2=np.linalg.norm(a-current)
            if dist2>total: dist1+=0
            else: dist1+=(1-(dist2/total))/len(self.targets)
        return dist1

class task7(task):
    """
    Make a tower with all the blocks (size particular order)
    """
    def generate(self,env):
        """
        Generate blocks in random places
        """
        sizes=[np.random.random() for i in range(3)]
        amount=np.random.randint(5,15)
        blocks=[]
        colour=[np.random.random(),np.random.random(),np.random.random(),1]
        min_dist = 0.15
        for i in range(amount):
            not_allowed=True 
            t1=time.time()
            t2=time.time()
            while not_allowed and t2-t1<2:
                t2=time.time()
                position = np.array([np.random.uniform(low=0.4, high=0.6), np.random.uniform(low=0.0, high=0.6), 0.05])
                # Check distances from all existing blocks
                if len(blocks) == 0:
                    blocks.append(position)
                    break
                distances = [np.linalg.norm(position[:2] - b[:2]) for b in np.array(blocks)]
                if all(d > min_dist for d in distances):
                    not_allowed=False
            if t2-t1<2:
                blocks.append(position)
                env.generate_block(position,colour,size=sizes[np.random.randint(0,3)])
            else:
                break
        env.step(100)
    def solve(self,env,p):
        point=np.array([np.random.uniform(low=0.4, high=0.5), np.random.uniform(low=-0.6, high=0.0), 0.10])
        targets=[]
        heights=[0,0,0]
        for i in range(3):
            targets.append(deepcopy(point))
            targets[i][0]+= 0.2 * i
        targetidx=-1
        id_=0
        category={}
        for i in range(len(env.block_ids)):
            #print("Correctness value:",self.get_correctness(env.get_observation()))
            visual_data = p.getVisualShapeData(env.block_ids[i])
            size=env.sizes[i]
            if category.get(size,-1) ==-1:
                category[size]=id_
                id_+=1
            targetidx=category[size]
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
        from sklearn.cluster import KMeans
        from sklearn.metrics import adjusted_rand_score
        #find the average groupings
        positions=obs['blocks']
        colours=obs['sizes']
 
        positions = np.array(positions)
        colours = np.array(colours, dtype=float)
        num_colours,colour_labels=np.unique(colours, axis=0, return_inverse=True)

        n_clusters=len(num_colours)
        if colours.max() > 1:  # normalize colour range
            colours /= 255.0

        kmeans = KMeans(n_clusters=n_clusters, n_init='auto').fit(positions)
        spatial_labels = kmeans.labels_
        centers = kmeans.cluster_centers_
        dists = np.linalg.norm(positions - centers[spatial_labels], axis=1)
        spatial_score = 1 - np.clip(np.mean(dists) / (np.max(dists) + 1e-9), 0, 1)
        alignment_score = adjusted_rand_score(colour_labels, spatial_labels)
        final_score = 0.6 * spatial_score + 0.4 * alignment_score
        return float(final_score)
    
class task8(task):
    """
    sort by size and colour in two axis
    """
    def generate(self,env):
        #generate random positions (maybe mroegrid like) with various sizes and colours
        m=np.random.randint(2,4)
        n=np.random.randint(2,4)
        position = np.array([np.random.uniform(low=0.35, high=0.5), np.random.uniform(low=0.0, high=0.6), 0.05])
        rgba=[0,0,0,1]
        channel=np.random.randint(0,3)
        offset=0.2
        for i in range(n):
            for j in range(m):
                temp=position.copy()
                temp[0]+=i/15 + np.random.uniform(low=-0.05,high=0.05) + offset
                temp[1]+=j/15 + np.random.uniform(low=-0.05,high=0.05) + offset
                size=np.random.uniform(low=0.3,high=1.5)
                shade=np.random.random()
                rgba[channel]=shade
                env.generate_block(temp,deepcopy(rgba),size=size) 
    def solve(self,env,p):
        sizes_idx=[]
        colour_idx=[]
        sizes=[]
        shades=[]
        #sort into size order
        for i in range(len(env.block_ids)):
            aabb_min, aabb_max = p.getAABB(env.block_ids[i])
            size = [aabb_max[i] - aabb_min[i] for i in range(3)][0] #enforce the first index as its a cube

            visual_data = p.getVisualShapeData(env.block_ids[i])
            colour=visual_data[0][7]
            grey_intensity=0.299*colour[0]+0.587*colour[1]+0.114*colour[2]
            placed=False 
            c=0
            while not placed and c<len(shades): #sort into colour order
                if grey_intensity<shades[c]:
                    shades.insert(c,grey_intensity)
                    colour_idx.insert(c,i)
                    placed=True
                c+=1
            if not placed:
                shades.append(grey_intensity)
                colour_idx.append(i)
            placed=False 
            c=0
            while not placed and c<len(sizes): ##sort into size
                if size<sizes[c]:
                    sizes.insert(c,size)
                    sizes_idx.insert(c,i)
                    placed=True
                c+=1
            if not placed:
                sizes.append(grey_intensity)
                sizes_idx.append(i)
        for i in range(len(env.block_ids)):#merge the orders in terms of x and y
            order_x=sizes_idx.index(i)/10 + 0.4 #start pos
            order_y=colour_idx.index(i)/10*-1
            print(order_x,order_y)
            target_pos=[order_x,order_y,0.05]
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
        #pick and place
        env.step(10000)
    def get_correctness(self,obs):
        #how in order the x is
        #how in order the y is 
        #add both and devide by two
        pass

if __name__=="__main__":
    from environment import *
    env=Env(realtime=0,speed=4)
    task=task8()
    task.generate(env)
    #env.record("/its/home/drs25/Documents/GitHub/Robot_shape_learning/Assets/Videos/task7_fast.mp4")
    task.save_details("/its/home/drs25/Documents/GitHub/Robot_shape_learning/Assets/Data/example.pkl",env)
    print("Correctness value:",task.get_correctness(env.get_observation()))
    task.solve(env,p)
    print("Correctness value:",task.get_correctness(env.get_observation()))
    #env.stop_record() 
    
    #test save
    env.close()
    del env 
    del task 
    env=Env(realtime=0)
    task=task8()
    task.load_details("/its/home/drs25/Documents/GitHub/Robot_shape_learning/Assets/Data/example.pkl",env)
    task.solve(env,p)