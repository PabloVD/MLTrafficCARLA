from collections import deque
import matplotlib.pyplot as plt
#from birdview2waymo import BirdviewBuffer
import carla
import numpy as np
from rasterizer import rasterize_input
import os
from roadgraph import RoadGraph

if not os.path.exists("testframes/"):
    os.system("mkdir testframes")
else:
    os.system("rm testframes/*")

n_channels = 11

#--- Constants
num_npcs = 9
prev_steps = n_channels

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
        print("npc not spawned")
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
waypoints = roadnet.roadpoints #map.generate_waypoints(0.9)
roads = np.array([[waypoint.transform.location.x, waypoint.transform.location.y] for waypoint in waypoints])
print(roads.shape)

# for npc in npcs:
#     npc.destroy()
# exit()

# print(roads[:10])
# print([[npc.get_transform().location.x, npc.get_transform().location.y] for npc in npcs])

world.tick()
frame_ind = 0

try:
    while True:

        for i, npc in enumerate(npcs):

            transf = npc.get_transform()

            # x, y, yaw, length, width
            agents_buffer_list[i].append([transf.location.x, transf.location.y, transf.rotation.yaw])
            #agents_buffer_list[i].append([transf.location.x, transf.location.y, transf.rotation.yaw, bb.extent.x*2, bb.extent.y*2])

        agents_arr = np.array(agents_buffer_list)

        if agents_arr.shape[1]==n_channels:
            
            raster = rasterize_input(agents_arr, bb_npcs, roads)
            np.save("testframes/test_"+str(frame_ind),raster[5])
            print(frame_ind, raster.shape)
            

            # Here will be the model
            # out = model(raster)

        world.tick()
        frame_ind+=1

finally:
    for npc in npcs:
        npc.destroy()



