import numpy as np

from PIL import Image
from shutil import rmtree
from pathlib import Path
import cv2
import os

image_folder = './image_output'
capture_fps = 25

class RobotRecorder:

    capture_i = 0
    

    def __init__(self):
        image_dir = Path(image_folder)
        if image_dir.exists():
            rmtree(image_dir)
        image_dir.mkdir()

    def capture_image(self, tensor):
        
        tensor = np.array(tensor.squeeze().cpu(), dtype=np.uint8)
        image = Image.fromarray(tensor)

        image.save(f"image_output/cam{self.capture_i:05}.png")

        self.capture_i += 1

    def create_video(self, video_name):
        # Compile simulation snapshots into a video
        
        # video_name = 'output_video.mp4'

        images = [img for img in os.listdir(image_folder) if img.endswith('.png')]
        images.sort() # Images get loaded out of order, need to organize them by name
        frame = cv2.imread(os.path.join(image_folder, images[0]))
        height, width, layers = frame.shape
        
        video = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'mp4v'), capture_fps, (width, height))
        
        for image in images:
            img_path = os.path.join(image_folder, image)
            frame = cv2.imread(img_path)
            video.write(frame)

        image_dir = Path(image_folder)
        if image_dir.exists():
            rmtree(image_dir)

        cv2.destroyAllWindows()
        video.release()