#--------------------------------------------
# Script to generate a video from a folder of frames
#--------------------------------------------

from moviepy.video.io import ImageSequenceClip
import glob
import os
from natsort import natsorted

#inpath = "testimages"
inpath = "trajectories/"

# Frame rate (Hz)
fps = 10

def get_video():

    # Get all frames in folder
    image_files = natsorted(glob.glob(inpath+"im_*"))

    # Create video and save
    clip = ImageSequenceClip.ImageSequenceClip(image_files, fps=fps)
    clip.write_videofile("video.mp4")


get_video()

