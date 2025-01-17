import os
import argparse
import cv2
import tensorflow as tf
from tqdm import tqdm
from moviepy import VideoFileClip
from transnetv2 import TransNetV2

# 加载模型
def load_model():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)

    model = tf.saved_model.load('inference/transnetv2-weights/')
    return model

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="视频场景切分工具")
    parser.add_argument("--video", required=True, help="输入视频路径")
    args = parser.parse_args()
    video_path = args.video

    video_name = os.path.basename(video_path)
    video_name_without_ext = os.path.splitext(video_name)[0]
    video_folder = os.path.dirname(video_path)
    output_folder = os.path.join(video_folder, video_name_without_ext)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 加载模型
    model = load_model()
    video_frames, single_frame_predictions, all_frame_predictions = model.predict_video(video_path)
    scenes = model.predictions_to_scenes(single_frame_predictions)

    video_clip = VideoFileClip(video_path)
    for i, (start, end) in enumerate(scenes):
        start_time = start / video_clip.fps
        end_time = end / video_clip.fps
        segment_clip = video_clip.subclip(start_time, end_time)
        output_path = os.path.join(output_folder, f'{video_name_without_ext}_{i + 1}.mp4')
        segment_clip.write_videofile(output_path, codec='libx264', fps=video_clip.fps)
    video_clip.close()

    input('\n任务已完成，按回车键退出……')
