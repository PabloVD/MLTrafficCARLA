from collections import deque
import matplotlib.pyplot as plt
#from birdview2waymo import BirdviewBuffer
import carla
import numpy as np
from rasterizer import rasterize_input
import os
from roadgraph import RoadGraph
import torch
from controller import VehiclePIDController

outpath = "testframes/"

if not os.path.exists(outpath):
    os.system("mkdir "+outpath)
else:
    os.system("rm "+outpath+"*")

class CustomWaypoint():
    def __init__(self, pos):
        self.transform = carla.Transform(location=carla.Location(x=pos[0],y=pos[1]))

def get_closest_waypoint(pos):
    loc = carla.Location(x=pos[0],y=pos[1])
    wp = map.get_waypoint(loc, project_to_road=True, lane_type=(carla.LaneType.Driving))
    return wp

device = "cuda"

# True for using NN, otherwise uses standard traffic manager
use_nn = True

# If True, uses PID to update position, otherwise teleports the vehicle
use_pid = False

# True for fixed birdview spectator, otherwise follows an agent
fixed_spec = True

n_channels = 11

# Timestep, 10Hz as Waymo data (do not modify)
dt = 0.1

#--- Constants
num_npcs = 9
prev_steps = n_channels

# Set simulation
client = carla.Client()
#client.load_world("Town03")
world = client.get_world()
map = world.get_map()
blueprint_library = world.get_blueprint_library()
settings = world.get_settings()
settings.synchronous_mode = True
settings.fixed_delta_seconds = dt
world.apply_settings(settings)
traffic_manager = client.get_trafficmanager()
traffic_manager.set_synchronous_mode(True)

# Spawn npcs
spawn_points = map.get_spawn_points()
blueprint_list = blueprint_library.filter("*audi*")
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
# list_wps = []
# for road in list_roads:
#     list_wps.extend(road)
# list_wps = [ [wp.x, wp.y] for wp in list_wps]

# Load model
model = torch.jit.load("models/model7.pt")
model = model.to(device)

# for npc in npcs:
#     npc.destroy()
# exit()

spec_offset = carla.Location(z=2,x=-2)

# Set spectator
if fixed_spec:
    spectator = world.get_spectator()
    specloc = carla.Location(z=220)
    specrot = carla.Rotation(pitch=-90)
    spectransf = carla.Transform(location=specloc, rotation=specrot)
    spectator.set_transform(spectransf)

# Set camera
# camera_bp = blueprint_library.find('sensor.camera.rgb')
# camera = world.spawn_actor(camera_bp, carla.Transform(carla.Location(x=50)), attach_to=spectator)
# camera.listen(lambda image: image.save_to_disk('_out/%06d.png' % image.frame))

# Traffic lights buffer
traffic_lights = world.get_actors().filter('traffic.traffic_light*')
tl_buffer_list = [deque(maxlen=prev_steps) for i in range(len(traffic_lights))]

# PID controllers
args_lateral_dict = {'K_P': 0., 'K_I': 0.05, 'K_D': 0.2, 'dt': dt}
args_longitudinal_dict = {'K_P': 1.0, 'K_I': 0.05, 'K_D': 0, 'dt': dt}
controllers = []
for npc in npcs:
    vehicle_controller = VehiclePIDController(npc,
                                            args_lateral=args_lateral_dict,
                                            args_longitudinal=args_longitudinal_dict,
                                            offset=0,
                                            max_throttle=0.75,
                                            max_brake=0.5,
                                            max_steering=0.8)
    controllers.append(vehicle_controller)

# Deactivate some layers for debugging
map_layer_names = [
            carla.MapLayer.Buildings,
            carla.MapLayer.Decals,
            carla.MapLayer.Foliage,
            carla.MapLayer.ParkedVehicles,
            carla.MapLayer.Props,
            carla.MapLayer.StreetLights,
            carla.MapLayer.Walls,
        ]

for layer in map_layer_names:
    world.unload_map_layer(layer)

min_frame = 100
run_nn = False

world.tick()
frame_ind = 0

print("Running with",len(npcs),"agents, total vehicles in simulation:", len(world.get_actors().filter('vehicle.*')))

try:
    while True:  

        # Attatch spectator to an actor
        if not fixed_spec:
            spectator = world.get_spectator()
            spectransf = npcs[0].get_transform()
            spectransf = carla.Transform(location=spectransf.location+spec_offset, rotation=spectransf.rotation)
            spectator.set_transform(spectransf)

        # Get agents information (x, y, yaw)
        for i, npc in enumerate(npcs):

            transf = npc.get_transform()
            agents_buffer_list[i].append([transf.location.x, transf.location.y, transf.rotation.yaw])

        # Get traffic light information
        for j, tl in enumerate(traffic_lights):
            transf = tl.get_transform()
            tl_buffer_list[j].append( [tl.get_state(), transf.location.x, transf.location.y] )

        # (N,timesteps,3)
        agents_arr = np.array(agents_buffer_list)

        if not run_nn and use_nn and agents_arr.shape[1]==n_channels and frame_ind>min_frame:
            print("Disabling traffic manager, switching to neural traffic")
            # for npc in npcs:
            #     npc.set_autopilot(False)
            #     control = carla.VehicleControl(steer=0., throttle=0., brake=1.0, hand_brake = False, manual_gear_shift = False)
            #     npc.apply_control(control)
            run_nn = True

        if run_nn:

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
                    
                    if use_pid:
                        # Apply PID controller

                        #target_vel = diffpos/dt
                        target_vel = (pred[1]-currpos)/(2.*dt)
                        target_speed = np.sqrt( target_vel[0]**2. + target_vel[1]**2. )
                        next_wp = get_closest_waypoint(pred[0])
                        #next_wp = CustomWaypoint(pred[0])
                        control = controllers[j].run_step(target_speed, next_wp)
                        npcs[j].apply_control(control)

                    else:
                        # Displace directly the vehicle to the predicted position

                        nextpos =  pred[0]

                        # Estimate orientation
                        diffpos = nextpos - currpos
                        newyaw = np.arctan2(diffpos[1], diffpos[0])*180./np.pi
                        
                        nextloc = carla.Location(x=nextpos[0], y=nextpos[1])
                        nextrot = carla.Rotation(yaw=newyaw)

                        npcs[j].set_transform(carla.Transform(location=nextloc, rotation=nextrot))

                np.save("testframes/prev_{:d}_{:03d}".format(j, frame_ind),agents_arr[j,:,:2])
                np.save("testframes/pred_{:d}_{:03d}".format(j, frame_ind),pred)

        world.tick()
        frame_ind+=1

finally:
    vehicles = world.get_actors().filter('vehicle.*')
    for vehicle in vehicles:
        vehicle.destroy()
    # for npc in npcs:
    #     npc.destroy()
    