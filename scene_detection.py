import tensorflow as tf
import cv2
import numpy as np
from transnetv2 import TransNetV2

class SceneDetector:
    def __init__(self):
        self.model = TransNetV2()
        
    def process_video(self, input_path, output_dir):
        # 读取视频
        video = cv2.VideoCapture(input_path)
        fps = video.get(cv2.CAP_PROP_FPS)
        frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # 获取场景切换点
        frames = []
        while video.isOpened():
            ret, frame = video.read()
            if not ret:
                break
            frames.append(frame)
            
        frames = np.array(frames)
        scene_scores = self.model.predict_raw(frames)
        scenes = self.model.predict_scenes(frames)
        
        # 保存切分后的场景
        for i, (start, end) in enumerate(scenes):
            scene_frames = frames[start:end]
            output_path = f"{output_dir}/scene_{i:04d}.mp4"
            self._save_video(scene_frames, output_path, fps)
            
        video.release()
        return scenes
        
    def _save_video(self, frames, output_path, fps):
        height, width = frames[0].shape[:2]
        writer = cv2.VideoWriter(
            output_path,
            cv2.VideoWriter_fourcc(*'mp4v'),
            fps,
            (width, height)
        )
        
        for frame in frames:
            writer.write(frame)
        writer.release()
