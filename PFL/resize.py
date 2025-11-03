import argparse
import os
import sys
from video_resizer import VideoResizer


# python crop.py --video_folder /test/Video --output_folder /test/cropped_video
def parse_arg_resize():
    parser = argparse.ArgumentParser(description="Crop all videos in a folder")
    parser.add_argument("--video_folder", type=str, required=True, help="Folder containing all videos")
    parser.add_argument("--output_folder", type=str, required=True, help="Output folder")
    parser.add_argument("--output_size", type=int, default=256, help="Size of the crop window")
    parser.add_argument("--gpu", type=str, default='0', help="GPU id")
    parser.add_argument("--max_workers", type=int, default=2, help="Number of workers")
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arg_resize()
    # 设置环境变量
    os.environ["OMP_NUM_THREADS"] = '16'
    # 设置GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    video_folder = args.video_folder
    output_folder = args.output_folder
    output_size = args.output_size
    max_workers = args.max_workers

    resizer = VideoResizer(video_folder, output_folder, output_size, max_workers)
    resizer.resize_folder_video()

