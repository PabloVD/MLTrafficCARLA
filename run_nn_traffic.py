from collections import deque
import matplotlib.pyplot as plt
import carla
import numpy as np
import os
import torch
import argparse

from rasterizer import rasterize_input
from roadgraph import RoadGraph
# from rasterizer_torch import get_rotation_matrix

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", type=str, required=True, help="Model name")
    parser.add_argument("-npcs", type=int, required=False, help="Number of NPCs", default=9)
    args = parser.parse_args()
    return args

class CustomWaypoint():
    def __init__(self, pos):
        self.transform = carla.Transform(location=carla.Location(x=pos[0],y=pos[1]))

def get_closest_waypoint(pos, map):
    loc = carla.Location(x=pos[0],y=pos[1])
    wp = map.get_waypoint(loc, project_to_road=True, lane_type=(carla.LaneType.Driving))
    return wp

def main():

    args = parse_args()

    outpath = "testframes/"

    if not os.path.exists(outpath):
        os.system("mkdir "+outpath)
    else:
        os.system("rm "+outpath+"*")

    # DEBUGGING
    outpath2 = "rendertests/"
    if not os.path.exists(outpath2):
        os.system("mkdir "+outpath2)
    else:
        os.system("rm "+outpath2+"*")

    logfile = os.getcwd()+"/logs/record.log"
    if not os.path.exists(os.getcwd()+"/logs/"):
        os.system("mkdir "+os.getcwd()+"/logs/")

    device = "cuda"

    # True for using NN, otherwise uses standard traffic manager
    use_nn = True

    # If True, uses PID to update position, otherwise teleports the vehicle
    use_pid = False

    # True for fixed birdview spectator, otherwise follows an agent
    fixed_spec = False#True

    # Number of previous timeframes taken as input
    prev_steps = 10

    # Zoom factor
    zoom_fact = 3#1.3

    # Timestep, 10Hz as Waymo data (do not modify)
    dt = 0.1

    # True for debugging, saves some debugging files and prints more information
    debug = False#True

    # Number of NPCs
    num_npcs = args.npcs

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
    bps = blueprint_library.filter("vehicle.*")
    blueprint_list = [x for x in bps if x.get_attribute('base_type') == 'car']
    # blueprint = blueprint_list[0]
    npcs = []
    for i in range(num_npcs):
        spwnpnt = spawn_points[i]
        #spwnpnt.location.x-=2
        blueprint = np.random.choice(blueprint_list)
        if blueprint.has_attribute('color'):
                color = np.random.choice(blueprint.get_attribute('color').recommended_values)
                blueprint.set_attribute('color', color)
        npc = world.try_spawn_actor(blueprint, spwnpnt)
        if not npc:
            print("npc "+str(i)+" not spawned")
        else:
            # Set Autopilot from standard Traffic Manager for initialization
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

    # Traffic lights
    traffic_lights = world.get_actors().filter('traffic.traffic_light*')

    # Road
    roadnet = RoadGraph(world)
    roadnet.get_tl_lanes(traffic_lights)

    # Load model
    print("Using model",args.m)
    model = torch.jit.load(args.m+".pt")
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

    # Frame when to switch to NN traffic manager
    min_frame = 100

    # Flag to use NN (False for initialization)
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
                agents_buffer_list[i].append([transf.location.x, transf.location.y, transf.rotation.yaw*np.pi/180.])
                
            # agents_arr: (N,timesteps,3)
            agents_arr = np.array(agents_buffer_list)

            # Get traffic light information
            tl_states = [ tl.get_state() for tl in traffic_lights ]

            if not run_nn and use_nn and agents_arr.shape[1]==prev_steps and frame_ind>min_frame:
                print("Disabling traffic manager, switching to neural traffic at frame "+str(frame_ind))
                run_nn = True
                for npc in npcs:
                    npc.set_autopilot(False)
                    control = carla.VehicleControl(steer=0., throttle=0., brake=1.0, hand_brake = False, manual_gear_shift = False)
                    npc.apply_control(control)
                client.start_recorder(logfile)

            # DEBUGGING
            # if frame_ind>10:
            #     raster = rasterize_input(agents_arr, bb_npcs, roadnet, tl_states, prev_steps, zoom_fact)
            #     raster = torch.tensor(raster, device=device, dtype=torch.float32)
            #     np.save(outpath2+"batch_torch"+str(frame_ind),raster[0].cpu().detach().numpy())


            if run_nn:

                if frame_ind>10:
                    raster = rasterize_input(agents_arr, bb_npcs, roadnet, tl_states, prev_steps, zoom_fact)
                    raster = torch.tensor(raster, device=device, dtype=torch.float32)
                #     # np.save(outpath2+"batch_torch"+str(frame_ind),raster[0].cpu().detach().numpy())

                # raster: (N,channels,rastersize,rastersize)
                # raster = rasterize_input(agents_arr, bb_npcs, roadnet, tl_states, prev_steps, zoom_fact)
                if debug:
                    np.save("testframes/test_"+str(frame_ind),raster[0])
                    # np.save("testframes/allagents_"+str(frame_ind),raster)
                    np.save("testframes/tlstates_"+str(frame_ind),[str(tl) for tl in tl_states])
                    print(frame_ind, agents_arr.shape, raster.shape)
                
                # Run model
                raster = torch.tensor(raster, device=device, dtype=torch.float32)
                # np.save(outpath2+"batch_torch"+str(frame_ind)+"_time_"+str(0),raster[0].cpu().detach().numpy())
                confidences, logits = model(raster)
                # print(confidences.shape, logits.shape)

                #currpos, yaw = agents_arr[:,-1,:2], agents_arr[:,-1,2]
                # Get current position and yaw
                currpos = torch.tensor(agents_arr[:,-1,:2], device=device, dtype=torch.float32)
                curryaw = torch.tensor(agents_arr[:,-1,2], device=device, dtype=torch.float32)
                nextpos, nextyaw = model.next_step(currpos, curryaw, confidences, logits)

                for j in range(len(agents_arr)):

                    # Store data and prediction for debugging
                    if debug:
                        np.save("testframes/prev_{:d}_{:03d}".format(j, frame_ind),agents_arr[j,:,:2])
                        # np.save("testframes/pred_{:d}_{:03d}".format(j, frame_ind),pred)


                    nextloc = carla.Location(x=nextpos[j,0].item(), y=nextpos[j,1].item())
                    nextrot = carla.Rotation(yaw=nextyaw[j].item()*180./np.pi)
                    # nextrot = carla.Rotation(yaw=curryaw[j].item()*180./np.pi)

                    npcs[j].set_transform(carla.Transform(location=nextloc, rotation=nextrot))


            world.tick()
            frame_ind+=1

    finally:
        vehicles = world.get_actors().filter('vehicle.*')
        for vehicle in vehicles:
            vehicle.destroy()
        
        client.stop_recorder()

        world.tick()
        print("Done")


if __name__=="__main__":

    main()