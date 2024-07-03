import numpy as np
import matplotlib.pyplot as plt
import glob
from natsort import natsorted
import cv2
from roadgraph import RoadGraph
from tqdm import tqdm
import os

get_roadmap = False

outpath = "trajectories/"

if not os.path.exists(outpath):
    os.system("mkdir "+outpath)
else:
    os.system("rm "+outpath+"*")

raster_size = 256
displacement = np.array([[raster_size // 2, raster_size // 2]])

# Map
if get_roadmap:

    import carla

    client = carla.Client()
    world = client.get_world()

    roadmap = np.ones((raster_size, raster_size, 3), dtype=np.uint8)*256

    roadnet = RoadGraph(world)
    waypointslist = roadnet.each_road_waypoints
    center = roadnet.center

    for waypoints in waypointslist:

        road = np.array([[waypoint.transform.location.x, waypoint.transform.location.y] for waypoint in waypoints])

        road = road - center + displacement

        roadmap = cv2.polylines(roadmap,[road.astype(int)],False,(0,0,0))

    np.save("roadmap",roadmap)
    np.save("center_roadmap",center)

else:
    roadmap = np.load("roadmap.npy")
    center = np.load("center_roadmap.npy")

roadmap = roadmap.transpose(1,0,2)

min_frame = 100
max_frame = len(glob.glob("testframes/prev_0_*.npy"))-1

for it, j in enumerate(tqdm(range(min_frame+1,min_frame+max_frame))):

    prevfil = natsorted(glob.glob("testframes/prev_*_{:03d}.npy".format(j)))
    predfil = natsorted(glob.glob("testframes/pred_*_{:03d}.npy".format(j)))

    plt.figure(figsize=(6,6),constrained_layout=True)
    plt.imshow(roadmap.astype(float)/256,alpha=0.5)

    for i in range(len(prevfil)):

        prev = np.load(prevfil[i])
        pred = np.load(predfil[i])

        prev = prev - center + displacement
        pred = pred - center + displacement
        pred = pred[:6]

        plt.scatter(prev[:,1],prev[:,0],color="b",s=5,alpha=0.7)
        plt.scatter(pred[:,1],pred[:,0],color="r",s=5,alpha=0.7)

    plt.xlim(0,raster_size)
    plt.ylim(0,raster_size)
    plt.title("Timeframe: {:.1f}s".format(it*0.1))
    plt.savefig(outpath+"im_"+str(j))
    plt.close()
