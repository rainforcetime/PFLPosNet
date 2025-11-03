import argparse
import os
import sys
from video_cropper import VideoCropper


# python crop.py --video_folder /test/Video --output_folder /test/cropped_video
def parse_arg_crop():
    parser = argparse.ArgumentParser(description="Crop all videos in a folder")
    parser.add_argument("--video_folder", type=str, required=True, help="Folder containing all videos")
    parser.add_argument("--output_folder", type=str, required=True, help="Output folder")
    parser.add_argument("--padding", type=int, default=20, help="Extra padding")
    parser.add_argument("--window_size", type=int, default=256, help="Size of the crop window")
    parser.add_argument("--gpu", type=str, default='0', help="GPU id")
    parser.add_argument("--max_workers", type=int, default=2, help="Number of workers")
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arg_crop()
    # 设置环境变量
    os.environ["OMP_NUM_THREADS"] = '16'
    # 设置GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    video_folder = args.video_folder
    output_folder = args.output_folder
    padding = args.padding
    window_size = args.window_size
    max_workers = args.max_workers

    cropper = VideoCropper(video_folder, output_folder, padding, window_size, max_workers)
    cropper.crop_folder_video()
