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

files = natsorted(glob.glob(inpath+"*"))

for j, file in enumerate(tqdm(files[:10])):

    raster = np.load(file)

    for i in range(n_channels):

        rast = raster[0:3].transpose(1, 2, 0).mean(-1)
        rast += raster[3+i]/2. + raster[3+11+i]  
        
        plt.imshow(rast)

        plt.savefig(outpath+"im_"+str(j)+"_"+str(i)+".png")

    # i = 10
    # rast = raster[0:3].transpose(1, 2, 0).mean(-1)
    # rast += raster[3+i]/2. + raster[3+11+i]  
    
    # plt.imshow(rast)

    # plt.savefig(outpath+"im_"+str(j)+".png")