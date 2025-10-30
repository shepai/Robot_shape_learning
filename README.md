This repo uses a robotic gym enivornment to explore approaches to robotic learning. 

## Simulation
The simulation uses pybullet, a 3D physics simulator made for python robotics integration. We make use of a default robot arm design and various 3D objects that are either in the deffault pybullet or the assets of this project. 

There are various tasks within the simulation such as building towers, organsing size/ colour etc... 

<table>
  <tr>
    <td align="center">
      <img src="Assets/Gifs/task1_fast.gif" width="250"><br>
      <b>Challenge 1</b>
    </td>
    <td align="center">
      <img src="Assets/Gifs/task2_fast.gif" width="250"><br>
      <b>Challenge 2</b>
    </td>
  </tr>
  <tr>
    <td align="center">
      <img src="Assets/Gifs/task3_fast.gif" width="250"><br>
      <b>Challenge 3</b>
    </td>
    <td align="center">
      <img src="Assets/Gifs/task4_fast.gif" width="250"><br>
      <b>Challenge 4</b>
    </td>
  </tr>
  <tr>
    <td align="center">
      <img src="Assets/Gifs/task5.gif" width="250"><br>
      <b>Challenge 5</b>
    </td>
    <td align="center">
      <img src="Assets/Gifs/task6_fast.gif" width="250"><br>
      <b>Challenge 6</b>
    </td>
  </tr>
</table>
We can create an environment by calling it, then generate random blocks, or generate specific blocks. We can also change the block urdf by giving it the filepath to other 3d models as long as it is in an pybullet accepted format ```blockname="cube_small.urdf"```. 

```python
env=Env()
env.generate_blocks(4) #four random blocks

position=[np.random.uniform(low=0.4, high=0.6), np.random.uniform(low=0.0, high=0.6), 0.05]
colour=[np.random.random(), np.random.random(), np.random.random(), 1]
env.generate_block(position,colour,size=1,blockname="cube_small.urdf") #generate an individual block
```
If you wish to record what is happening yyou can use the record function.

```python 
env.record("/pathto/video_example.mp4")
#do something
env.stop_record()
```

The simulator can be moved using various commands such as ```move_gripper_to(coordinate)``` . We can get coordinates of the blocks from where each block id is stored in the aray ```env.block_ids``` as an array. Picking up a block requires the block id. 

For example if we were to place a block on top of another block we could manipulate the robot using these commands and block ids.

```python

cube_pos, _ = p.getBasePositionAndOrientation(env.block_ids[0]) #get the first block
cube_pos=list(cube_pos)
cube_pos[2]+=0.08 #go above it so it doesn't get crushed
env.move_gripper_to(cube_pos) #move the robot arm to the top of it
#time.sleep(2)
env.pick_block(env.block_ids[0]) #pick it up using the magnet style robot arm
up=np.array(cube_pos)
up[2]+=0.6
env.move_gripper_to(up) #move to a point that is above
cube_pos, _ = p.getBasePositionAndOrientation(env.block_ids[1])
cube_pos=np.array(cube_pos)
cube_pos[2]+=0.18
env.move_gripper_to(cube_pos)
env.put_block() #drop the object
env.move_gripper_to(up)

```

## Dataset creation 
We use these tasks to make a dataset that shows the task in the start and end. The challenge is making the robot go from start to end. The state of the enironment can be saved using the following code:

```python
env=Env(realtime=0)
task=task2() #example task 
task.generate(env)
task.save_details("file/path/starting_state.pkl",env) #this 
task.solve(env,p)
task.save_details("file/path/end_state.pkl",env) #this 
env.close()
```

We can then load the positions using:

```python
env=Env(realtime=0)
task=task2() #example task
task.load_details("/its/home/drs25/Documents/GitHub/Robot_shape_learning/Assets/Data/example.pkl",env)
task.solve(env,p)
```


## Dependiencies 
We make use of the PyBullet robot arm m
```
pip install pybullet
```

## Learning
It is broken up into varipous parts, the parent model, the child model and the simulation itself. The parent child model is under development but a potential idea for how these tasks could be solved. 

### Parent model 

### Child Model 

