import matplotlib.pyplot as plt
import numpy as np
from natsort import natsorted
import glob
import os
from tqdm import tqdm

n_channels = 11

inpath="testframes/"
outpath="testimages/"

if not os.path.exists(outpath):
    os.system("mkdir "+outpath)
else:
    os.system("rm "+outpath+"*")

files = natsorted(glob.glob(inpath+"test_*"))

# raster expected in shape (imgsize,imgsize,channels)
def raster2rgb(raster, i):

    road = raster[:,:,0:3]
    img = np.copy(road)
    ego = raster[:,:,3+i]
    others = raster[:,:,3+11+i]

    ego = ego[:,:,None]
    others = others[:,:,None]

    zeros = np.zeros_like(ego)
    ego_pos = np.concatenate([ego,ego,ego],axis=-1)
    ego = np.concatenate([zeros,zeros,ego],axis=-1)
    others_pos = np.concatenate([others,others,others],axis=-1)
    others = np.concatenate([zeros,others,zeros],axis=-1)/2.

    img[ego_pos!=0]=ego[ego_pos!=0]
    img[others_pos!=0]=others[others_pos!=0]

    return img


for j, file in enumerate(tqdm(files)):
    raster = np.load(file)
    #raster = raster.transpose(2, 1, 0) 
    raster = raster.transpose(1, 2, 0)

    #for i in range(n_channels):
    for i in [10]:

        img = raster2rgb(raster, i)

        plt.imshow(img)
        plt.axis('off')

        plt.savefig(outpath+"im_"+str(j)+"_"+str(i)+".png", bbox_inches='tight')
