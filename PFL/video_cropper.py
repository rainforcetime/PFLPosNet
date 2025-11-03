import glob
import os

import cv2
from tqdm import tqdm
import concurrent.futures
from multiprocessing import Value, Manager


class VideoCropper(object):
    def __init__(self, input_folder, output_folder, padding=20, window_size=256, max_workers=2, tag='v1', face_minSize=60, face_minNeighbors=6, tar_size=256):
        # self.video_path = video_path
        self.input_folder = input_folder
        self.output_folder = output_folder
        self.padding = padding
        self.window_size = window_size
        self.tar_size = tar_size
        self.max_workers = max_workers
        self.tag = tag
        self.face_minSize = face_minSize
        self.face_minNeighbors = face_minNeighbors
        self.face_scaleFactor = 1.1
        # 初始化边界值
        self.left = 2000  # 使用正无穷大表示最左端初始值
        self.top = 2000  # 使用正无穷大表示最上端初始值
        self.right = 0  # 使用负无穷大表示最右端初始值
        self.bottom = 0  # 使用负无穷大表示最下端初始值

        # # 加载预训练的人脸检测分类器
        # self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    def crop_folder_video(self):
        """
        从视频文件夹中裁剪所有视频, 并将其保存到输出文件夹
        如果input_folder中递归包含子文件夹，则会处理所有子文件夹中的视频，并将其保存到output_folder中的相应子文件夹中
        :param input_folder: 输入的视频文件夹
        :param output_folder:  输出的视频文件夹
        :param padding:  额外的边距
        :param window_size:  裁剪窗口的大小
        :return:  None
        """
        # 获取所有视频文件的路径
        video_files = glob.glob(os.path.join(self.input_folder, '**/*.mp4'), recursive=True)

        # 排除已经裁剪过的视频
        video_files = [video_file for video_file in video_files if not os.path.exists(os.path.join(self.output_folder, os.path.relpath(video_file, self.input_folder)))]

        # 使用 ProcessPoolExecutor 来并行处理视频文件
        with concurrent.futures.ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            list(tqdm(executor.map(self._crop_video_wrapper, video_files), total=len(video_files)))

        print("All videos have been cropped.")

    def re_crop_folder_video(self, check_path):
        # 获取所有视频文件的路径
        video_files = glob.glob(os.path.join(self.input_folder, '**/*.mp4'), recursive=True)
        # 排除已经处理过的视频文件
        video_files = [video_file for video_file in video_files if not os.path.exists(
            os.path.join(check_path, os.path.relpath(video_file, self.input_folder)).replace('.mp4', '.npy'))]

        print("没有处理的文件数量: ", len(video_files))

        file_name = f'without_deal_{self.tag}.txt'

        # 将其写入文件
        with open(file_name, 'w') as f:
            for file in video_files:
                f.write(file + '\n')
        print(f"没有处理的文件已经写入到{file_name}")

        # 使用 ProcessPoolExecutor 来并行处理视频文件
        with concurrent.futures.ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            list(tqdm(executor.map(self._crop_video_wrapper, video_files), total=len(video_files)))

        print("All videos have been cropped.")


    def _crop_video_wrapper(self, video_file):
        """
        包装器函数，用于并行处理单个视频文件
        :param video_file: 视频文件路径
        :return: 是否成功裁剪
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
        return self.crop_video(video_file, output_file)

    def crop_video(self, input_file, out_file):
        """
        裁剪视频
        :param input_file: 输入视频文件路径
        :param out_file: 输出视频文件路径
        :param padding: 额外的边距
        :param window_size: 裁剪窗口的大小
        :return: 是否成功裁剪 True/False
        """
        # 第一步：找到人脸的边界
        left, top, right, bottom = self.find_face_bounds_byHaar_3(input_file)
        # 第二步：根据边界裁剪视频
        is_success = self.crop_video_by_bound(input_file, out_file, left, top, right, bottom)

        return is_success

    def find_face_bounds_byHaar(self, input_file):
        """
        使用 OpenCV 的 Haar 级联检测器找到视频中人脸的边界
        TODO 优化代码，减少重复代码
        :param input_file: 输入视频文件路径
        :param padding: 额外的边距
        :param window_size: 裁剪窗口的大小
        :return: 人脸的边界值
        """
        # 加载预训练的人脸检测分类器
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

        # 打开视频文件
        video = cv2.VideoCapture(input_file)

        # 获取视频的宽度和高度
        frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # 初始化边界值
        left = self.left
        top = self.top
        right = self.right
        bottom = self.bottom

        # 计算视频的总帧数
        total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

        # # 创建进度条
        # progress_bar = tqdm(total=total_frames, desc="Finding Face Bounds")

        prev_left, prev_top, prev_right, prev_bottom = left, top, right, bottom  # 上一帧的人脸边界

        while True:
            ret, frame = video.read()

            if not ret:
                break

            # 更新进度条
            # progress_bar.update(1)
            # 在进度条显示当前left, top, right, bottom
            # progress_bar.set_postfix(left=left, top=top, right=right, bottom=bottom)

            # 转换为灰度图像
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # 检测人脸
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=6, minSize=(60, 60))

            if len(faces) > 0:
                # 选择与上一帧边界位置最接近的人脸
                min_distance = float('inf')
                selected_face = None

                for (fx, fy, fw, fh) in faces:
                    # 计算当前人脸的边界
                    this_left = fx
                    this_top = fy
                    this_right = fx + fw
                    this_bottom = fy + fh

                    # 计算当前人脸与上一帧边界之间的距离
                    distance = abs(this_left - prev_left) + abs(this_top - prev_top) + abs(
                        this_right - prev_right) + abs(this_bottom - prev_bottom)

                    # 选择距离最小的人脸
                    if distance < min_distance:
                        min_distance = distance
                        selected_face = (fx, fy, fw, fh)

                if selected_face is not None:
                    # 使用选定的人脸更新边界值
                    (fx, fy, fw, fh) = selected_face
                    left = min(left, fx)
                    top = min(top, fy)
                    right = max(right, fx + fw)
                    bottom = max(bottom, fy + fh)

                    # 更新上一帧的边界
                    prev_left, prev_top, prev_right, prev_bottom = left, top, right, bottom

                # 如果没有检测到人脸，则保留上一帧的边界
                elif selected_face is None:
                    left, top, right, bottom = prev_left, prev_top, prev_right, prev_bottom

                # print("left, top, right, bottom:", left, top, right, bottom)

        # # 关闭进度条
        # progress_bar.close()

        # 释放资源
        video.release()

        # 添加额外的边距
        left = max(0, left - self.padding)
        top = max(0, top - self.padding)
        right = min(frame_width, right + self.padding)
        bottom = min(frame_height, bottom + self.padding)

        # 使宽度和高度相同，以便裁剪为正方形，将宽度和高度设置为最大值
        width = right - left
        height = bottom - top
        max_side = max(width, height)

        # 如果宽度或高度超过窗口大小，则进行均匀收缩
        if max_side > self.window_size:
            # 计算需要收缩的比例
            ratio = self.window_size / max_side
            # 根据比例调整边界值
            left = int(left + (width - width * ratio) / 2)
            top = int(top + (height - height * ratio) / 2)
            right = int(left + self.window_size)
            bottom = int(top + self.window_size)

        # 调整边界值以确保裁剪部分为300x300
        if right - left != self.window_size or bottom - top != self.window_size:
            # 计算中心点
            center_x = (left + right) / 2
            center_y = (top + bottom) / 2
            # 重新计算边界值
            left = int(center_x - self.window_size / 2)
            top = int(center_y - self.window_size / 2)
            right = int(center_x + self.window_size / 2)
            bottom = int(center_y + self.window_size / 2)

            # 确保边界值在图像范围内
            if left < 0:
                left = 0
                right = self.window_size
            if top < 0:
                top = 0
                bottom = self.window_size
            if right > frame_width:
                right = frame_width
                left = frame_width - self.window_size
            if bottom > frame_height:
                bottom = frame_height
                top = frame_height - self.window_size

        self.left = left
        self.top = top
        self.right = right
        self.bottom = bottom

        # 返回边界值
        return self.left, self.top, self.right, self.bottom


    def find_face_bounds_byHaar_2(self, input_file):
        """
        使用 OpenCV 的 Haar 级联检测器找到视频中人脸的边界
        策略二，平均边界扩张
        :param input_file: 输入视频文件路径
        :param padding: 额外的边距
        :param window_size: 裁剪窗口的大小
        :return: 人脸的边界值
        """
        # 加载预训练的人脸检测分类器
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

        # 打开视频文件
        video = cv2.VideoCapture(input_file)

        # 获取视频的宽度和高度
        frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

        left = self.left
        top = self.top
        right = self.right
        bottom = self.bottom

        # 初始化边界值
        all_left = []
        all_top = []
        all_right = []
        all_bottom = []

        # 计算视频的总帧数
        total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

        # # 创建进度条
        # progress_bar = tqdm(total=total_frames, desc="Finding Face Bounds")

        prev_left, prev_top, prev_right, prev_bottom = left, top, right, bottom  # 上一帧的人脸边界

        while True:
            ret, frame = video.read()

            if not ret:
                break

            # 更新进度条
            # progress_bar.update(1)
            # 在进度条显示当前left, top, right, bottom
            # progress_bar.set_postfix(left=left, top=top, right=right, bottom=bottom)

            # 转换为灰度图像
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # 检测人脸
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=6, minSize=(60, 60))

            if len(faces) > 0:
                # 选择与上一帧边界位置最接近的人脸
                min_distance = float('inf')
                selected_face = None

                for (fx, fy, fw, fh) in faces:
                    # 计算当前人脸的边界
                    this_left = fx
                    this_top = fy
                    this_right = fx + fw
                    this_bottom = fy + fh

                    # 计算当前人脸与上一帧边界之间的距离
                    distance = abs(this_left - prev_left) + abs(this_top - prev_top) + abs(
                        this_right - prev_right) + abs(this_bottom - prev_bottom)

                    # 选择距离最小的人脸
                    if distance < min_distance:
                        min_distance = distance
                        selected_face = (fx, fy, fw, fh)

                if selected_face is not None:
                    # 使用选定的人脸更新边界值
                    (fx, fy, fw, fh) = selected_face
                    all_left.append(fx)
                    all_top.append(fy)
                    all_right.append(fx + fw)
                    all_bottom.append(fy + fh)

                    # 更新上一帧的边界
                    prev_left, prev_top, prev_right, prev_bottom = fx, fy, fx + fw, fy + fh

                # 如果没有检测到人脸，则保留上一帧的边界
                elif selected_face is None:
                    all_left.append(prev_left)
                    all_top.append(prev_top)
                    all_right.append(prev_right)
                    all_bottom.append(prev_bottom)
                    # left, top, right, bottom = prev_left, prev_top, prev_right, prev_bottom

                # print("left, top, right, bottom:", left, top, right, bottom)

        # # 关闭进度条
        # progress_bar.close()

        # 释放资源
        video.release()

        # 排除len(all_left) == 0的情况
        if len(all_left) == 0:
            return 0, 0, self.window_size, self.window_size
        if len(all_top) == 0:
            return 0, 0, self.window_size, self.window_size
        if len(all_right) == 0:
            return 0, 0, self.window_size, self.window_size
        if len(all_bottom) == 0:
            return 0, 0, self.window_size, self.window_size

        # 计算平均边界
        left = int(sum(all_left) / len(all_left))
        top = int(sum(all_top) / len(all_top))
        right = int(sum(all_right) / len(all_right))
        bottom = int(sum(all_bottom) / len(all_bottom))


        # 添加额外的边距
        left = max(0, left - self.padding)
        top = max(0, top - self.padding)
        right = min(frame_width, right + self.padding)
        bottom = min(frame_height, bottom + self.padding)

        # 使宽度和高度相同，以便裁剪为正方形，将宽度和高度设置为最大值
        width = right - left
        height = bottom - top
        max_side = max(width, height)

        # 如果宽度或高度超过窗口大小，则进行均匀收缩
        if max_side > self.window_size:
            # 计算需要收缩的比例
            ratio = self.window_size / max_side
            # 根据比例调整边界值
            left = int(left + (width - width * ratio) / 2)
            top = int(top + (height - height * ratio) / 2)
            right = int(left + self.window_size)
            bottom = int(top + self.window_size)

        # 调整边界值以确保裁剪部分为300x300
        if right - left != self.window_size or bottom - top != self.window_size:
            # 计算中心点
            center_x = (left + right) / 2
            center_y = (top + bottom) / 2
            # 重新计算边界值
            left = int(center_x - self.window_size / 2)
            top = int(center_y - self.window_size / 2)
            right = int(center_x + self.window_size / 2)
            bottom = int(center_y + self.window_size / 2)

            # 确保边界值在图像范围内
            if left < 0:
                left = 0
                right = self.window_size
            if top < 0:
                top = 0
                bottom = self.window_size
            if right > frame_width:
                right = frame_width
                left = frame_width - self.window_size
            if bottom > frame_height:
                bottom = frame_height
                top = frame_height - self.window_size

        self.left = left
        self.top = top
        self.right = right
        self.bottom = bottom

        # 返回边界值
        return self.left, self.top, self.right, self.bottom

    def find_face_bounds_byHaar_3(self, input_file):
        """
        使用 OpenCV 的 Haar 级联检测器找到视频中人脸的边界
        策略二，平均边界扩张
        :param input_file: 输入视频文件路径
        :param padding: 额外的边距
        :param window_size: 裁剪窗口的大小
        :return: 人脸的边界值
        """
        # 加载预训练的人脸检测分类器
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

        # 打开视频文件
        video = cv2.VideoCapture(input_file)

        # 获取视频的宽度和高度
        frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

        left = self.left
        top = self.top
        right = self.right
        bottom = self.bottom

        # 初始化边界值
        all_left = []
        all_top = []
        all_right = []
        all_bottom = []

        # 计算视频的总帧数
        total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

        # # 创建进度条
        # progress_bar = tqdm(total=total_frames, desc="Finding Face Bounds")

        prev_left, prev_top, prev_right, prev_bottom = left, top, right, bottom  # 上一帧的人脸边界
        loc = 0

        while True:
            ret, frame = video.read()

            if not ret:
                break

            # 更新进度条
            # progress_bar.update(1)
            # 在进度条显示当前left, top, right, bottom
            # progress_bar.set_postfix(left=left, top=top, right=right, bottom=bottom)

            # 转换为灰度图像
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # 检测人脸
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=6, minSize=(60, 60))

            if len(faces) > 0:
                # 选择与上一帧边界位置最接近的人脸
                min_distance = float('inf')
                selected_face = None
                face_num = 0

                min_top = float('inf')

                for (fx, fy, fw, fh) in faces:
                    # 计算当前人脸的边界
                    this_left = fx
                    this_top = fy
                    this_right = fx + fw
                    this_bottom = fy + fh
                    face_num += 1
                    # print(f"第{loc}帧的第{face_num}个人脸边界: left={this_left}, top={this_top}, right={this_right}, bottom={this_bottom}")

                    # 计算当前人脸与上一帧边界之间的距离
                    distance = abs(this_left - prev_left) + abs(this_top - prev_top) + abs(
                        this_right - prev_right) + abs(this_bottom - prev_bottom)


                    # 如果是第一帧，则选择fy最小的人脸
                    if loc == 0:
                        if fy < min_top:
                            min_top = fy
                            selected_face = (fx, fy, fw, fh)
                    else:
                        # 选择距离最小的人脸
                        if distance < min_distance:
                            min_distance = distance
                            selected_face = (fx, fy, fw, fh)

                if selected_face is not None:
                    # 使用选定的人脸更新边界值
                    (fx, fy, fw, fh) = selected_face
                    all_left.append(fx)
                    all_top.append(fy)
                    all_right.append(fx + fw)
                    all_bottom.append(fy + fh)

                    # 更新上一帧的边界
                    prev_left, prev_top, prev_right, prev_bottom = fx, fy, fx + fw, fy + fh

                # 如果没有检测到人脸，则保留上一帧的边界
                elif selected_face is None:
                    all_left.append(prev_left)
                    all_top.append(prev_top)
                    all_right.append(prev_right)
                    all_bottom.append(prev_bottom)
                    # left, top, right, bottom = prev_left, prev_top, prev_right, prev_bottom
            loc += 1

                # print("left, top, right, bottom:", left, top, right, bottom)

        # # 关闭进度条
        # progress_bar.close()

        # 释放资源
        video.release()

        # 排除len(all_left) == 0的情况
        if len(all_left) == 0:
            return 0, 0, self.window_size, self.window_size
        if len(all_top) == 0:
            return 0, 0, self.window_size, self.window_size
        if len(all_right) == 0:
            return 0, 0, self.window_size, self.window_size
        if len(all_bottom) == 0:
            return 0, 0, self.window_size, self.window_size

        # 计算平均边界
        left = int(sum(all_left) / len(all_left))
        top = int(sum(all_top) / len(all_top))
        right = int(sum(all_right) / len(all_right))
        bottom = int(sum(all_bottom) / len(all_bottom))


        # 添加额外的边距
        left = max(0, left - self.padding)
        top = max(0, top - self.padding)
        right = min(frame_width, right + self.padding)
        bottom = min(frame_height, bottom + self.padding)

        # 使宽度和高度相同，以便裁剪为正方形，将宽度和高度设置为最大值
        width = right - left
        height = bottom - top
        max_side = max(width, height)

        # 如果宽度或高度超过窗口大小，则进行均匀收缩
        if max_side > self.window_size:
            # 计算需要收缩的比例
            ratio = self.window_size / max_side
            # 根据比例调整边界值
            left = int(left + (width - width * ratio) / 2)
            top = int(top + (height - height * ratio) / 2)
            right = int(left + self.window_size)
            bottom = int(top + self.window_size)

        # 调整边界值以确保裁剪部分为300x300
        if right - left != self.window_size or bottom - top != self.window_size:
            # 计算中心点
            center_x = (left + right) / 2
            center_y = (top + bottom) / 2
            # 重新计算边界值
            left = int(center_x - self.window_size / 2)
            top = int(center_y - self.window_size / 2)
            right = int(center_x + self.window_size / 2)
            bottom = int(center_y + self.window_size / 2)

            # 确保边界值在图像范围内
            if left < 0:
                left = 0
                right = self.window_size
            if top < 0:
                top = 0
                bottom = self.window_size
            if right > frame_width:
                right = frame_width
                left = frame_width - self.window_size
            if bottom > frame_height:
                bottom = frame_height
                top = frame_height - self.window_size

        self.left = left
        self.top = top
        self.right = right
        self.bottom = bottom

        # 返回边界值
        return self.left, self.top, self.right, self.bottom

    def crop_video_by_bound(self, input_file, output_file, left, top, right, bottom):
        """
        根据给定的边界值裁剪视频
        :param input_file:  输入视频文件路径
        :param output_file:  输出视频文件路径
        :param left:  左边界
        :param top:  上边界
        :param right:  右边界
        :param bottom:  下边界
        :return:  裁剪是否成功 True/False
        """
        # 打开视频文件
        video = cv2.VideoCapture(input_file)

        # 获取视频的帧率
        fps = video.get(cv2.CAP_PROP_FPS)

        # 获取视频的宽度和高度
        frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # 创建 VideoWriter 对象以保存输出视频
        out = cv2.VideoWriter(output_file, cv2.VideoWriter_fourcc(*'mp4v'), fps, (self.tar_size, self.tar_size))

        # 一次性读取整个视频
        frames = []
        while True:
            ret, frame = video.read()
            if not ret:
                break
            frames.append(frame)

        # 关闭视频文件
        video.release()

        # 裁剪所有帧
        cropped_frames = [f[top:bottom, left:right] for f in frames]

        # Resize 视频帧
        if self.tar_size != self.window_size:
            cropped_frames = [cv2.resize(frame, (self.tar_size, self.tar_size)) for frame in cropped_frames]

        # 写入裁剪后的帧
        for frame in cropped_frames:
            out.write(frame)

        # 释放资源
        out.release()
        # TODO 细化返回值
        if len(cropped_frames) == 0:
            return False
        else:
            return True