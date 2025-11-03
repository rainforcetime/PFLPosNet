import glob
import os

import cv2
from tqdm import tqdm
import concurrent.futures


class VideoResizer(object):
    def __init__(self, input_folder, output_folder, output_size=256, max_workers=2):
        self.input_folder = input_folder
        self.output_folder = output_folder
        self.output_size = output_size
        self.max_workers = max_workers

    def resize_folder_video(self):
        """
        从视频文件夹中resize所有视频, 并将其保存到输出文件夹
        如果input_folder中递归包含子文件夹，则会处理所有子文件夹中的视频，并将其保存到output_folder中的相应子文件夹中
        :return:
        """
        # 获取所有视频文件的路径
        video_files = glob.glob(os.path.join(self.input_folder, '**/*.mp4'), recursive=True)

        # 排除已经resize过的视频
        video_files = [video_file for video_file in video_files if not os.path.exists(
            os.path.join(self.output_folder, os.path.relpath(video_file, self.input_folder)))]

        # 使用 ProcessPoolExecutor 来并行处理视频文件
        with concurrent.futures.ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            list(tqdm(executor.map(self._resize_video_wrapper, video_files), total=len(video_files)))

        print("All videos have been resized.")

    def _resize_video_wrapper(self, video_file):
        """
        用于包装resize_video方法的方法
        :param video_file: 视频文件路径
        :return: None
        """
        # 获取视频文件的相对路径
        relative_path = os.path.relpath(video_file, self.input_folder)

        # 获取视频文件的文件夹路径
        folder_path = os.path.dirname(relative_path)

        # 创建输出文件夹
        os.makedirs(os.path.join(self.output_folder, folder_path), exist_ok=True)

        # 构建输出文件路径
        output_file = os.path.join(self.output_folder, relative_path)

        # 裁剪视频
        return self.resize_video(video_file, output_file)

    def resize_video(self, input_file, out_file):
        """
        resize视频
        :param input_file: 输入视频文件路径
        :param out_file:  输出视频文件路径
        :return:
        """
        # 读取视频
        video = cv2.VideoCapture(input_file)

        # 获取视频的帧率
        fps = video.get(cv2.CAP_PROP_FPS)

        # 获取视频的宽度和高度
        width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

        output_size = (self.output_size, self.output_size)

        # 创建VideoWriter对象
        out = cv2.VideoWriter(out_file, cv2.VideoWriter_fourcc(*'mp4v'), fps, output_size)

        # 读取视频帧
        while True:
            ret, frame = video.read()
            if not ret:
                break

            # resize帧
            frame = cv2.resize(frame, output_size)

            # 写入帧
            out.write(frame)

        # 释放资源
        video.release()
        out.release()

        return True
