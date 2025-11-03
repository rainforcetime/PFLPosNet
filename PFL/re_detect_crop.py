import argparse
import os
import sys
from video_cropper import VideoCropper


# python re_detect_crop.py --video_folder ./test/need_re_crop --output_folder ./test/need_re_crop_output
# python re_detect_crop.py --video_folder ./test/need_re_crop_test --output_folder ./test/need_re_crop_output_test
def parse_arg():
    parser = argparse.ArgumentParser(description="Detect Crop all videos in a folder")
    parser.add_argument("--video_folder", type=str, required=True, help="Folder containing all videos")
    parser.add_argument("--output_folder", type=str, required=True, help="Output folder")
    parser.add_argument("--padding", type=int, default=20, help="Extra padding")
    parser.add_argument("--window_size", type=int, default=256, help="Size of the crop window")
    parser.add_argument("--gpu", type=str, default='2', help="GPU id")
    parser.add_argument("--max_workers", type=int, default=2, help="Number of workers")
    parser.add_argument("--face_minSize", type=int, default=60, help="face_minSize")
    parser.add_argument("--face_minNeighbors", type=int, default=6, help="face_minNeighbors")

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arg()
    # 设置环境变量
    os.environ["OMP_NUM_THREADS"] = '16'
    # 设置GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    video_folder = args.video_folder
    output_folder = args.output_folder
    padding = args.padding
    window_size = args.window_size
    max_workers = args.max_workers
    face_minSize = args.face_minSize
    face_minNeighbors = args.face_minNeighbors

    cropper = VideoCropper(video_folder, output_folder, padding, window_size, max_workers,
                           face_minSize, face_minNeighbors)
    cropper.crop_folder_video()
