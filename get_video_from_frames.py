#--------------------------------------------
# Script to generate a video from a folder of frames
#--------------------------------------------

from moviepy.video.io import ImageSequenceClip
import glob
from natsort import natsorted
import argparse

def argument_parser():

    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        '--images',
        default='out',
        help='Folder where frames are stored')
    argparser.add_argument(
        '--video',
        default="video",
        type=str,
        help='Name of the output video')
    argparser.add_argument(
        '--fps',
        default=10,
        type=int,
        help='FPS')
    
    return argparser.parse_args()

args = argument_parser()

image_folder = args.images
video_name = args.video+'.mp4'
fps=args.fps

# Get all frames in folder
image_files = natsorted(glob.glob(image_folder+"/*.png"))
image_files = image_files[:-1]

# Create video and save
clip = ImageSequenceClip.ImageSequenceClip(image_files, fps=fps)
clip.write_videofile(video_name)