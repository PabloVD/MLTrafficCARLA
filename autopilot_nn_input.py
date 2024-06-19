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

offset = carla.Location(z=2,x=-2)

# Set specator
spectator = world.get_spectator()
specloc = carla.Location(z=200)
specrot = carla.Rotation(pitch=-90)
spectransf = carla.Transform(location=specloc, rotation=specrot)
spectator.set_transform(spectransf)

try:
    while True:

        # Attatch spectator to an actor
        # spectator = world.get_spectator()
        # spectransf = npcs[0].get_transform()
        # spectransf = carla.Transform(location=spectransf.location+offset, rotation=spectransf.rotation)
        # spectator.set_transform(spectransf)

        for i, npc in enumerate(npcs):

            transf = npc.get_transform()

            # x, y, yaw
            agents_buffer_list[i].append([transf.location.x, transf.location.y, transf.rotation.yaw])

        # (N,timesteps,3)
        agents_arr = np.array(agents_buffer_list)

        if agents_arr.shape[1]==n_channels and frame_ind>20:

            # (N,channels,rastersize,rastersize)
            raster = rasterize_input(agents_arr, bb_npcs, list_roads)
            np.save("testframes/test_"+str(frame_ind),raster[0])
            print(frame_ind, agents_arr.shape, raster.shape)
            
            # Run model
            raster = torch.tensor(raster, device=device, dtype=torch.float32)
            confidences, logits  = model(raster)
            # print(confidences.shape, logits.shape)

            #currpos, yaw = agents_arr[:,-1,:2], agents_arr[:,-1,2]
            
            # TO DO improve with array multiplication

            for j in range(len(agents_arr)):

                pred = logits[j,confidences[j].argmax()].detach().cpu().numpy()

                currpos, yaw = agents_arr[j,-1,:2], agents_arr[j,-1,2]*np.pi/180.

                rot_matrix = np.array([
                    [np.cos(-yaw), -np.sin(-yaw)],
                    [np.sin(-yaw), np.cos(-yaw)],
                ])
            
                pred = pred@rot_matrix + currpos 

                if use_nn:

                    nextpos =  pred[0]

                    # Estimate orientation
                    diffpos = nextpos - currpos
                    newyaw = np.arctan2(diffpos[1],diffpos[0])*180./np.pi
                    
                    nextloc = carla.Location(x=nextpos[0],y=nextpos[1])
                    nextrot = carla.Rotation(yaw=newyaw)

                    npcs[j].set_transform(carla.Transform(location=nextloc,rotation=nextrot))

                np.save("testframes/prev_"+str(j)+"_"+str(frame_ind),agents_arr[j,:,:2])
                np.save("testframes/pred_"+str(j)+"_"+str(frame_ind),pred)

        world.tick()
        frame_ind+=1

finally:
    vehicles = world.get_actors().filter('vehicle.*')
    for vehicle in vehicles:
        vehicle.destroy()
    # for npc in npcs:
    #     npc.destroy()
    