import argparse
import glob
import os

from video_cropper import VideoCropper


def parse_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, help='Path to the file')
    parser.add_argument('--res_folder', type=str, help='Path to the output file')
    parser.add_argument('--check_path', type=str, help='Path to the check file')
    parser.add_argument('--tag', type=str, help='output file tag')
    parser.add_argument('--gpu', type=str, default='1', help='GPU id')
    return parser.parse_args()

# python check_recrop_video.py --input ./test/Video --check_path ./test/3dmm_v1 --res_folder ./test/cropped_video_360 --tag v1

if __name__ == '__main__':
    args = parse_arg()

    # without_files = find_without_deal_npy(args.input, args.res_folder)
    # print("没有处理的文件数量: ", len(without_files))
    #
    # file_name = f'without_deal_{args.tag}.txt'
    #
    # # 将其写入文件
    # with open(file_name, 'w') as f:
    #     for file in without_files:
    #         f.write(file + '\n')
    # print(f"没有处理的文件已经写入到{file_name}")

    # 设置环境变量
    os.environ["OMP_NUM_THREADS"] = '16'
    # 设置GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    video_folder = args.input
    output_folder = args.res_folder
    padding = 20
    window_size = 360
    max_workers = 2

    cropper = VideoCropper(video_folder, output_folder, padding, window_size, max_workers)
    cropper.re_crop_folder_video(args.check_path)
