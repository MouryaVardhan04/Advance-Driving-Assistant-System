"""
lane_detector_env

python main.py 

Lane Lines Detection pipeline

Usage:
    main.py [INPUT_PATH]

Options:

-h --help show this screen
"""

import numpy as np
import matplotlib.image as mpimg
import cv2
from docopt import docopt
import pygame
from moviepy.editor import VideoFileClip
from CameraCalibration import CameraCalibration
from Thresholding import *
from PerspectiveTransformation import *
from LaneLines import *

class FindLaneLines:
    """ This class is for parameter tunning.

    Attributes:
        ...
    """
    def __init__(self):
        """ Init Application"""
        self.calibration = CameraCalibration('camera_cal', 9, 6)
        self.thresholding = Thresholding()
        self.transform = PerspectiveTransformation()
        self.lanelines = LaneLines()

    def forward(self, img):
        out_img = np.copy(img)
        img = self.calibration.undistort(img)
        img = self.transform.forward(img)
        img = self.thresholding.forward(img)
        img = self.lanelines.forward(img)
        img = self.transform.backward(img)

        out_img = cv2.addWeighted(out_img, 1, img, 0.6, 0)
        out_img = self.lanelines.plot(out_img)
        return out_img

    def process_image(self, input_path):
        img = mpimg.imread(input_path)
        out_img = self.forward(img)
        return out_img

    def process_video(self, input_path):
        clip = VideoFileClip(input_path)
        
        # Initialize Pygame
        pygame.init()
        screen_width = clip.size[0]
        screen_height = clip.size[1]
        screen = pygame.display.set_mode((screen_width, screen_height))
        pygame.display.set_caption("Lane Detection")
        
        # Process each frame
        for frame in clip.iter_frames():
            out_frame = self.forward(frame)
            
            # Convert the frame to Pygame format
            frame_rgb = cv2.cvtColor(out_frame, cv2.COLOR_BGR2RGB)
            frame_pygame = pygame.surfarray.make_surface(frame_rgb.swapaxes(0, 1))
            
            # Display the frame
            screen.blit(frame_pygame, (0, 0))
            pygame.display.flip()
            
            # Handle events (e.g., closing the window)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return  # Exit the function

        # Clean up Pygame
        pygame.quit()


def main():
    args = docopt(__doc__)
    
    # Corrected: Use the names from the Usage string: 'INPUT_PATH'
    input_path = args['INPUT_PATH'] if args['INPUT_PATH'] else 'project_video.mp4'

    findLaneLines = FindLaneLines()
    
    findLaneLines.process_video(input_path)


if __name__ == "__main__":
    main()