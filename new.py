import os

from moviepy.editor import VideoFileClip
from transnetv2 import TransNetV2

if __name__ == '__main__':

    video_path = input('请输入视频文件路径\n')
    while not os.path.isfile(video_path):
        video_path = input('请输入正确的视频文件路径\n')
    video_name = os.path.basename(video_path)
    video_name_without_ext = os.path.splitext(video_name)[0]
    video_folder = os.path.dirname(video_path)
    output_folder = os.path.join(video_folder, video_name_without_ext)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    model = TransNetV2()
    video_frames, single_frame_predictions, all_frame_predictions = model.predict_video_2(video_path)
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
