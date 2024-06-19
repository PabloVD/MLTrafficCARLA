import numpy as np
import matplotlib.pyplot as plt
import glob
from natsort import natsorted
import carla
import cv2
from roadgraph import RoadGraph
from tqdm import tqdm
import os

outpath = "trajectories/"

if not os.path.exists(outpath):
    os.system("mkdir "+outpath)
else:
    os.system("rm "+outpath+"*")

# Map
raster_size = 224
displacement = np.array([[raster_size // 2, raster_size // 2]])

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

maxframe = len(glob.glob("testframes/prev_0_*.npy"))-1

# Agents
npcs = 9

for it, j in enumerate(tqdm(range(21,maxframe))):

    prevfil = natsorted(glob.glob("testframes/prev_*"+str(j)+".npy"))
    predfil = natsorted(glob.glob("testframes/pred_*"+str(j)+".npy"))

    plt.figure(figsize=(6,6),constrained_layout=True)
    plt.imshow(roadmap.astype(float)/256,alpha=0.5)

    for i in range(len(prevfil)):

        prev = np.load(prevfil[i])
        pred = np.load(predfil[i])

        prev = prev - center + displacement
        pred = pred - center + displacement
        pred = pred[:10]

        plt.scatter(prev[:,0],prev[:,1],color="b",s=5,alpha=0.7)
        plt.scatter(pred[:,0],pred[:,1],color="r",s=5,alpha=0.7)

    plt.xlim(0,raster_size)
    plt.ylim(0,raster_size)
    plt.title("Timeframe: {:.1f}s".format(it*0.1))
    plt.savefig(outpath+"im_"+str(j))
    plt.close()
