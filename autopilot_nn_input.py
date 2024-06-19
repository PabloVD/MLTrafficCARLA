from collections import deque
import matplotlib.pyplot as plt
#from birdview2waymo import BirdviewBuffer
import carla
import numpy as np
from rasterizer import rasterize_input
import os
from roadgraph import RoadGraph
import torch

if not os.path.exists("testframes/"):
    os.system("mkdir testframes")
else:
    os.system("rm testframes/*")

device = "cuda"

use_nn = True

n_channels = 11

#--- Constants
num_npcs = 9
prev_steps = n_channels

# Set simulation
client = carla.Client()
world = client.get_world()
map = world.get_map()
settings = world.get_settings()
settings.synchronous_mode = True
settings.fixed_delta_seconds = 0.1  # 10Hz as Waymo data
world.apply_settings(settings)
traffic_manager = client.get_trafficmanager()
traffic_manager.set_synchronous_mode(True)

# Spawn npcs
spawn_points = map.get_spawn_points()
blueprint_list = world.get_blueprint_library().filter("*audi*")
blueprint = blueprint_list[0]
npcs = []
for i in range(num_npcs):
    npc = world.try_spawn_actor(blueprint, spawn_points[i])
    if not npc:
        print("npc "+str(i)+" not spawned")
    else:
        npc.set_autopilot(True)
        npcs.append( npc )

# Length and width bounding boxes
bb_npcs = []
for npc in npcs:
    bb = npc.bounding_box
    bb_npcs.append([bb.extent.x*2, bb.extent.y*2])
bb_npcs = np.array(bb_npcs)

# Create buffer with previous steps info
agents_buffer_list = [deque(maxlen=prev_steps) for i in range(len(npcs))]

# Road
roadnet = RoadGraph(world)
list_roads = roadnet.each_road_waypoints

# Load model
model = torch.jit.load("model.pt")
model = model.to(device)

# for npc in npcs:
#     npc.destroy()
# exit()

world.tick()
frame_ind = 0

try:
    while True:

        for i, npc in enumerate(npcs):

            transf = npc.get_transform()

            # x, y, yaw, length, width
            agents_buffer_list[i].append([transf.location.x, transf.location.y, transf.rotation.yaw])

        # (N,timesteps,3)
        agents_arr = np.array(agents_buffer_list)

        if agents_arr.shape[1]==n_channels and frame_ind>30:

            # (N,channels,rastersize,rastersize)
            raster = rasterize_input(agents_arr, bb_npcs, list_roads)
            np.save("testframes/test_"+str(frame_ind),raster[5])
            print(frame_ind, agents_arr.shape, raster.shape)
            
            # Run model
            raster = torch.tensor(raster, device=device, dtype=torch.float32)
            confidences, logits  = model(raster)
            #print(confidences.shape, logits.shape)

            
            # rot_matrix = np.array(
            # [
            #     [np.cos(-yaw), -np.sin(-yaw)],
            #     [np.sin(-yaw), np.cos(-yaw)],
            # ]
            # )
            pred = logits[np.arange(len(logits)),confidences.argmax(axis=1)]
            
            #pred = pred@rot_matrix + shift 

        world.tick()
        frame_ind+=1

finally:
    for npc in npcs:
        npc.destroy()



