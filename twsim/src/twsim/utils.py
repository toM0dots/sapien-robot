from pathlib import Path
from shutil import rmtree

import cv2
import numpy as np


class RobotRecorder:
    capture_i = 0

    def __init__(self, output_dir, fps, overwrite):
        self.image_dir = Path(output_dir)
        self.fps = fps

        if self.image_dir.is_file():
            raise ValueError(f"{self.image_dir} is a file, not a directory.")

        if self.image_dir.exists() and not overwrite:
            raise ValueError(f"{self.image_dir} already exists")

        if self.image_dir.exists() and overwrite:
            rmtree(self.image_dir)

        self.image_dir.mkdir(parents=True, exist_ok=True)

    def capture_image(self, tensor):
        image_as_array = np.array(tensor.squeeze().cpu(), dtype=np.uint8)
        image_filepath = self.image_dir / f"{self.capture_i:05}.png"
        cv2.imwrite(str(image_filepath), image_as_array)

        self.capture_i += 1

    def save_as_video(self, video_name, overwrite):
        if Path(video_name).is_file() and not overwrite:
            raise ValueError(f"{video_name} already exists")

        images = sorted(self.image_dir.glob("*.png"))
        if not images:
            raise ValueError("No images found in the output directory.")

        first_frame = cv2.imread(str(images[0]))
        height, width, _ = first_frame.shape

        fourcc = cv2.VideoWriter.fourcc(*"mp4v")
        video = cv2.VideoWriter(video_name, fourcc, self.fps, (width, height))

        for image_path in images:
            frame = cv2.imread(str(image_path))
            video.write(frame)

        video.release()
