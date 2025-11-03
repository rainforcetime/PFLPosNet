import os
import time

import numpy as np
import torch
import argparse
from tqdm import tqdm
import logging

from dataset.dataset import get_dataloader
from utils.render import Render
from metric import *
from utils.util import load_config, init_seed, get_logging_path, get_epoch_num_from_path
import model as module_arch


# python evaluate_only_render_pos.py --gpu 6
def parse_arg():
    parser = argparse.ArgumentParser(description="PyTorch Training")
    # for loading the trained-model weights.
    parser.add_argument("--epoch_num", type=int, help="epoch number of saving model weight", default=30)
    parser.add_argument("--exp_num", type=int, help="the number of training experiment.", default=11)
    parser.add_argument("--mode", type=str, help="train (val) or test", default="test")
    # parser.add_argument("--config", type=str, help="config path", default="./config/train_diffusion_past_sliding_short_pre.yaml")
    parser.add_argument("--config", type=str, help="config path", default="./config/pos/base.yaml")
    # parser.add_argument("--config", type=str, help="config path", default="./config/past_sliding_short_divide_all.yaml")
    # parser.add_argument("--config", type=str, help="config path", default="./config/train_diffusion.yaml")
    # parser.add_argument("--config", type=str, help="config path", default="./config/train_diffusion_150_past.yaml")
    parser.add_argument("--evaluate_log_dir", type=str, default="./log/evaluate")  # evaluate
    parser.add_argument('--test_period', type=int, default=10)
    parser.add_argument("--gpu", type=int, help="gpu id", default=4)
    parser.add_argument("--tag", type=str, help="tag", default="")
    args = parser.parse_args()
    return args


def compute_mse(prediction, target):
    _, k, seq_len, dim = prediction.shape
    # join last two dimensions of prediction and target
    prediction = prediction.reshape(-1, k, seq_len * dim)
    target = target.reshape(-1, k, seq_len * dim)
    loss = ((prediction - target) ** 2).mean(axis=-1)  # (batch_size, k)
    return torch.mean(loss)


def evaluate(cfg, device, model, test_loader, render, split):
    model.eval()

    iteration = 0
    epoch_num = get_epoch_num_from_path(cfg.trainer.resume)
    out_dir = os.path.join(cfg.trainer.out_dir, split, 'exp_' + str(cfg.exp_num), 'epoch_' + str(epoch_num))
    if cfg.tag != "":
        out_dir = os.path.join(out_dir, cfg.tag)
    os.makedirs(out_dir, exist_ok=True)


    for batch_idx, (
            speaker_audio_clip,
            speaker_video_clip,
            speaker_emotion_clip,
            speaker_3dmm_clip,
            listener_video_clip,
            listener_emotion_clip,
            listener_3dmm_clip,
            listener_3dmm_clip_personal,
            listener_reference,
    ) in enumerate(tqdm(test_loader)):
        _3dmm_dim = listener_3dmm_clip.shape[-1]

        (speaker_audio_clip,  # (bs, token_len, 78)
         speaker_video_clip,  # (bs, token_len, 3, 224, 224)
         speaker_emotion_clip,  # (bs, token_len, 25)
         speaker_3dmm_clip,  # (bs, token_len, 58)
         listener_video_clip,  # (bs, token_len, 3, 224, 224)
         listener_emotion_clip,  # (bs, token_len, 25)
         listener_3dmm_clip,  # (bs, token_len, 58)
         listener_3dmm_clip_personal,  # (bs * k, token_len, 58)
         listener_reference) = \
            (speaker_audio_clip.to(device),
             speaker_video_clip.to(device),
             speaker_emotion_clip.to(device),
             speaker_3dmm_clip.to(device),
             listener_video_clip.to(device),
             listener_emotion_clip.to(device),
             listener_3dmm_clip.to(device),
             listener_3dmm_clip_personal.to(device),
             listener_reference.to(device))  # (bs, 3, 224, 224)

        # listener_3dmm_clip_save = listener_3dmm_clip.detach().clone().cpu()
        # just for dimension compatibility during inference
        listener_emotion_clip = listener_emotion_clip.repeat_interleave(
            cfg.test_dataset.k_appro, dim=0)  # (bs * k, token_len, 25)
        listener_3dmm_clip = listener_3dmm_clip.repeat_interleave(
            cfg.test_dataset.k_appro, dim=0)  # (bs * k, token_len, 58)

        # if (batch_idx % cfg.test_period) == 0:
        with torch.no_grad():
            input_dict = {
                "speaker_audio": speaker_audio_clip,
                "speaker_emotion_input": speaker_emotion_clip,
                "speaker_3dmm_input": speaker_3dmm_clip,
                "listener_emotion_input": listener_emotion_clip,
                "listener_3dmm_input": listener_3dmm_clip,
                "listener_personal_input": listener_3dmm_clip_personal,
                "listener_reference": listener_reference,
            }

            _, listener_3dmm_pred, _ = model(x=input_dict)
            # listener_3dmm_pred["prediction_3dmm"].shape: (bs, k_appro==10, seq_len==750, 3dmm_dim==58)
            listener_3dmm_pred = listener_3dmm_pred["prediction_3dmm"]


            # TODO: Rendering!
            render.rendering_for_mp4(
                out_dir,
                "{}_iter_{}".format(split, str(batch_idx + 1)),
                listener_3dmm_pred[0, 1],  # (750, 58)
                speaker_video_clip[0],  # (750, 3, 224, 224)
                listener_reference[0],  # (3, 224, 224)
                listener_video_clip[0],  # (750, 3, 224, 224)
                step=1,  # set frame step to 10.
            )



def main(args):
    gpu_id = args.gpu
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)  # Make only the first GPU visible
    os.environ['OMP_NUM_THREADS'] = '8'  # For debugging
    # load yaml config
    cfg = load_config(args=args, config_path=args.config)
    print("cfg: ", args.config)
    init_seed(seed=cfg.trainer.seed)  # seed initialization

    # logging
    logging_path = get_logging_path(cfg.evaluate_log_dir, cfg.exp_num)
    os.makedirs(logging_path, exist_ok=True)
    logging.basicConfig(filename=logging_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')

    test_loader = get_dataloader(cfg.test_dataset)
    # test_loader = get_dataloader(cfg.validation_dataset)
    split = cfg.test_dataset.split

    # Set device ordinal if GPUs are available
    if torch.cuda.device_count() > 0:
        device = torch.device('cuda:0')  # Adjust the device ordinal as needed
        render = Render('cuda')
    else:
        device = torch.device('cpu')
        render = Render()

    diff_model = getattr(module_arch, cfg.trainer.model)(cfg, device)
    diff_model = diff_model.to(device)

    pos_model = getattr(module_arch, cfg.pos_model.type)(cfg, diff_model, device)
    pos_model.to(device)


    logging.info("-----------------Start Rendering-----------------")

    evaluate(cfg, device, pos_model, test_loader, render, split)  # render

    logging.info("-----------------Finish Rendering-----------------")


if __name__ == '__main__':
    main(args=parse_arg())
