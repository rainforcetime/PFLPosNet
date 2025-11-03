import os
import logging
import time
import datetime
import torch
import argparse
from torch import optim
from functools import partial
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm

from dataset.dataset import get_dataloader
from dataset.dataset_short import get_dataloader_short
import model.diffusion.utils.losses as module_loss
from torch.utils.tensorboard import SummaryWriter
from utils.util import load_config, init_seed, get_logging_path, get_tensorboard_path, AverageMeter, save_checkpoint, \
    get_lr, collect_grad_stats, optimizer_resume
import model as module_arch
import random
import numpy as np
# from utils.render import Render

# os.environ['CUDA_VISIBLE_DEVICES'] = '4'  # Make only the first GPU visible
# # os.environ['CUDA_LAUNCH_BLOCKING'] = '1'  # For debugging

# python train_main_warmup.py --exp_num 2225 --config ./config/baseline_smo.yaml --gpu 6
def parse_arg():
    parser = argparse.ArgumentParser(description="PyTorch Training")
    parser.add_argument("--exp_num", type=int, help="the number of the experiment.", default=18903)
    parser.add_argument("--mode", type=str, help="train (val) or test", default="train")
    # parser.add_argument("--writer", type=bool, help="whether use tensorboard", required=True)
    parser.add_argument("--writer", type=bool, help="whether use tensorboard", default=True)
    # parser.add_argument("--config", type=str, help="config path", default="./config/train_diffusion_past_sliding_short_pre.yaml")
    # parser.add_argument("--config", type=str, help="config path", default="./config/train_diffusion_past_sliding_short.yaml")
    parser.add_argument("--config", type=str, help="config path", default="./config/exp/18903.yaml")
    # parser.add_argument("--config", type=str, help="config path", default="./config/train_diffusion_short.yaml")
    # parser.add_argument("--config", type=str, help="config path", default="./config/past_sliding_short_divide_all.yaml")
    parser.add_argument("--gpu", type=int, help="gpu id", default=6)
    args = parser.parse_args()
    return args


def train(cfg, model, train_loader, optimizer, scheduler, criterion, epoch, writer, device):
    whole_losses = AverageMeter()
    diff_prior_losses = AverageMeter()
    diff_decoder_losses = AverageMeter()
    temporal_losses = AverageMeter()
    smooth_losses = AverageMeter()
    pos_losses = AverageMeter()
    trans_losses = AverageMeter()


    # 使用 tqdm 包装 train_loader，并在描述中显示当前的 epoch
    progress_bar = tqdm(train_loader, desc=f'exp_{cfg.exp_num} Epoch {epoch + 1}/{cfg.trainer.epochs}', leave=True)

    model.train()
    for batch_idx, (
            speaker_audio_clip,
            speaker_video_clip,  # (bs, token_len, 3, 224, 224)
            speaker_emotion_clip,
            speaker_3dmm_clip,
            listener_video_clip,  # (bs, token_len, 3, 224, 224)
            listener_emotion_clip,
            listener_3dmm_clip,
            listener_3dmm_clip_personal,
            listener_reference,
    ) in enumerate(progress_bar):

        batch_size = speaker_audio_clip.shape[0]
        (speaker_audio_clip,  # (bs, token_len, 78)
         speaker_emotion_clip,  # (bs, token_len, 25)
         speaker_3dmm_clip,  # (bs, token_len, 58)
         listener_emotion_clip,  # (bs * k, token_len, 25)
         listener_3dmm_clip,  # (bs * k, token_len, 58)
         listener_3dmm_clip_personal,  # (bs * k, token_len, 58)
         listener_reference) = \
            (speaker_audio_clip.to(device),
             speaker_emotion_clip.to(device),
             speaker_3dmm_clip.to(device),
             listener_emotion_clip.to(device),
             listener_3dmm_clip.to(device),
             listener_3dmm_clip_personal.to(device),
             listener_reference.to(device))  # (bs, 3, 224, 224)

        output_prior, output_decoder = model(
            speaker_audio=speaker_audio_clip,
            speaker_emotion_input=speaker_emotion_clip,
            speaker_3dmm_input=speaker_3dmm_clip,
            listener_emotion_input=listener_emotion_clip,
            listener_3dmm_input=listener_3dmm_clip,
            listener_personal_input=listener_3dmm_clip_personal,
        )
        # output_prior['encoded_prediction'].shape: [bs, k_appro, 1, 512]
        # output_prior['encoded_target'].shape: [bs, k_appro, 1, 512]
        # output_decoder['prediction_3dmm'].shape: [bs, k_appro, window_size, 58]
        # output_decoder['target_3dmm'].shape: [bs, k_appro, window_size, 58]

        # TODO: debug: save 3dmm predictions and GT
        with torch.no_grad():
            if batch_idx in [0, 1] and (epoch + 1) % 20 == 0:
                pred = output_decoder['prediction_3dmm'].detach().cpu().numpy()
                gt = output_decoder['target_3dmm'].detach().cpu().numpy()
                save_dir = os.path.join(cfg.trainer.out_dir, 'train', 'exp_' + str(cfg.exp_num), 'save_3dmm')
                os.makedirs(save_dir, exist_ok=True)
                np.save(os.path.join(save_dir, 'epoch_{}_iter_{}_pred_3dmm.npy'.format((epoch + 1), batch_idx)), pred)
                np.save(os.path.join(save_dir, 'epoch_{}_iter_{}_gt_3dmm.npy'.format((epoch + 1), batch_idx)), gt)

        output = criterion(output_prior, output_decoder)
        loss = output["loss"]  # whole training loss
        temporal_loss = output["temporal_loss"]  # temporal constraints
        diff_prior_loss = output["encoded"]
        diff_decoder_loss = output["decoded"]
        if cfg.loss.type in ["DiffusionSmoothLoss", "DiffusionSmoothPosLoss", "DiffusionSmoothTransLoss"]:
            smooth_loss = output["smooth_loss"]

        if cfg.loss.type in ["DiffusionSmoothPosLoss"]:
            pos_loss = output["pos_loss"]

        if cfg.loss.type in ["DiffusionSmoothTransLoss"]:
            trans_loss = output["trans_loss"]

        iteration = batch_idx + len(train_loader) * epoch

        if writer is not None:
            writer.add_scalar("Train/whole_loss", loss.data.item(), iteration)
            writer.add_scalar("Train/diff_prior_loss", diff_prior_loss.data.item(), iteration)
            writer.add_scalar("Train/diff_decoder_loss", diff_decoder_loss.data.item(), iteration)
            writer.add_scalar("Train/temporal_loss", temporal_loss.data.item(), iteration)
            if cfg.loss.type in ["DiffusionSmoothLoss", "DiffusionSmoothPosLoss", "DiffusionSmoothTransLoss"]:
                writer.add_scalar("Train/smooth_loss", smooth_loss.data.item(), iteration)

            if cfg.loss.type in ["DiffusionSmoothPosLoss"]:
                writer.add_scalar("Train/pos_loss", pos_loss.data.item(), iteration)

            if cfg.loss.type in ["DiffusionSmoothTransLoss"]:
                writer.add_scalar("Train/trans_loss", trans_loss.data.item(), iteration)



        whole_losses.update(loss.data.item(), batch_size)
        diff_prior_losses.update(diff_prior_loss.data.item(), batch_size)
        diff_decoder_losses.update(diff_decoder_loss.data.item(), batch_size)
        temporal_losses.update(temporal_loss.data.item(), batch_size)
        if cfg.loss.type in ["DiffusionSmoothLoss", "DiffusionSmoothPosLos", "DiffusionSmoothTransLoss"]:
            smooth_losses.update(smooth_loss.data.item(), batch_size)

        if cfg.loss.type in ["DiffusionSmoothPosLoss"]:
            pos_losses.update(pos_loss.data.item(), batch_size)

        if cfg.loss.type in ["DiffusionSmoothTransLoss"]:
            trans_losses.update(trans_loss.data.item(), batch_size)

        optimizer.zero_grad()
        loss.backward()

        if cfg.trainer.clip_grad:
            clip_grad_norm_(parameters=model.parameters(), max_norm=5, norm_type=2)
        optimizer.step()

    # warmup for the first 5 epochs
    if scheduler is not None and (epoch + 1) >= 5:
        scheduler.step()

    # obtain the learning rate
    lr = get_lr(optimizer=optimizer)
    if writer is not None:
        writer.add_scalar("Train/lr", lr, epoch)

    if cfg.loss.type in ["DiffusionSmoothLoss"]:
        return (whole_losses.avg, diff_prior_losses.avg, diff_decoder_losses.avg,
                temporal_losses.avg, smooth_losses.avg)

    if cfg.loss.type in ["DiffusionSmoothPosLoss"]:
        return (whole_losses.avg, diff_prior_losses.avg, diff_decoder_losses.avg,
                temporal_losses.avg, smooth_losses.avg, pos_losses.avg)

    if cfg.loss.type in ["DiffusionSmoothTransLoss"]:
        return (whole_losses.avg, diff_prior_losses.avg, diff_decoder_losses.avg,
                temporal_losses.avg, smooth_losses.avg, trans_losses.avg)

    return whole_losses.avg, diff_prior_losses.avg, diff_decoder_losses.avg, temporal_losses.avg


def validate(cfg, model, val_loader, criterion, epoch, device):
    whole_losses = AverageMeter()
    diff_prior_losses = AverageMeter()
    diff_decoder_losses = AverageMeter()
    temporal_losses = AverageMeter()

    # 使用 tqdm 包装 train_loader，并在描述中显示当前的 epoch
    progress_bar = tqdm(val_loader, desc=f'exp_{cfg.exp_num} Epoch {epoch + 1}/{cfg.trainer.epochs}', leave=True)

    model.eval()
    for batch_idx, (
            speaker_audio_clip,
            _,  # speaker_video_clip
            speaker_emotion_clip,
            speaker_3dmm_clip,
            _,  # listener_video_clip
            listener_emotion_clip,
            listener_3dmm_clip,
            listener_3dmm_clip_personal,
            _,  # listener_reference
    ) in enumerate(progress_bar):
        batch_size = speaker_audio_clip.shape[0]
        (speaker_audio_clip,  # (bs, token_len, 78)
         speaker_emotion_clip,  # (bs, token_len, 25)
         speaker_3dmm_clip,  # (bs, token_len, 58)
         listener_emotion_clip,  # (bs * k, token_len, 25)
         listener_3dmm_clip,  # (bs * k, token_len, 58)
         # (bs * k, token_len, 58)
         listener_3dmm_clip_personal) = \
            (speaker_audio_clip.to(device),
             speaker_emotion_clip.to(device),
             speaker_3dmm_clip.to(device),
             listener_emotion_clip.to(device),
             listener_3dmm_clip.to(device),
             listener_3dmm_clip_personal.to(device))

        with torch.no_grad():
            output_prior, output_decoder = model(
                speaker_audio=speaker_audio_clip,
                speaker_emotion_input=speaker_emotion_clip,
                speaker_3dmm_input=speaker_3dmm_clip,
                listener_emotion_input=listener_emotion_clip,
                listener_3dmm_input=listener_3dmm_clip,
                listener_personal_input=listener_3dmm_clip_personal,
            )

            output = criterion(output_prior, output_decoder)
            loss = output["loss"]  # whole training loss
            temporal_loss = output["temporal_loss"]  # temporal constraints
            diff_prior_loss = output["encoded"]
            diff_decoder_loss = output["decoded"]

            whole_losses.update(loss.data.item(), batch_size)
            diff_prior_losses.update(diff_prior_loss.data.item(), batch_size)
            diff_decoder_losses.update(diff_decoder_loss.data.item(), batch_size)
            temporal_losses.update(temporal_loss.data.item(), batch_size)

    return whole_losses.avg, diff_prior_losses.avg, diff_decoder_losses.avg, temporal_losses.avg


def main(args):
    gpu_id = args.gpu
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)  # Make only the first GPU visible
    os.environ['OMP_NUM_THREADS'] = '8'  # For debugging
    # load yaml config
    cfg = load_config(args=args, config_path=args.config)
    print("config: ", args.config)
    init_seed(seed=cfg.trainer.seed)  # seed initialization
    # lowest_val_loss = 10000

    # logging
    logging_path = get_logging_path(cfg.trainer.log_dir, cfg.exp_num)
    os.makedirs(logging_path, exist_ok=True)
    logging.basicConfig(filename=logging_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')

    if cfg.writer:
        writer_path = get_tensorboard_path(cfg.trainer.tb_dir, cfg.exp_num)
        writer = SummaryWriter(writer_path)
    else:
        writer = None

    train_loader = get_dataloader(cfg.dataset)
    # val_loader = get_dataloader(cfg.validation_dataset)

    # Set device ordinal if GPUs are available
    if torch.cuda.device_count() > 0:
        device = torch.device('cuda:0')  # Adjust the device ordinal as needed
    else:
        device = torch.device('cpu')
    model = getattr(module_arch, cfg.trainer.model)(cfg, device)
    model.to(device)

    if cfg.optimizer.type == "adamW":
        optimizer = optim.AdamW(model.parameters(), betas=cfg.optimizer.args.beta, lr=cfg.optimizer.args.lr,
                                weight_decay=cfg.optimizer.args.weight_decay)
    elif cfg.optimizer == "adam":
        optimizer = optim.Adam(model.parameters(), cfg.optimizer.args.lr, weight_decay=cfg.optimizer.args.weight_decay)
    elif cfg.optimizer == "sgd":
        optimizer = optim.SGD(model.parameters(), cfg.optimizer.args.lr, momentum=cfg.optimizer.args.momentum,
                              weight_decay=cfg.optimizer.args.weight_decay)
    else:
        NotImplemented("The optimizer {} not implemented.".format(cfg.optimizer.type))

    if cfg.trainer.get("resume", None) is not None:
        optimizer_resume(cfg, model, optimizer, device)

    criterion = partial(getattr(module_loss, cfg.loss.type), **cfg.loss.args)

    if cfg.optimizer.scheduler == "cosine_annealing":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_loader), eta_min=0,
                                                               last_epoch=-1)
    else:
        scheduler = None

    # log the config file
    logging.info("config: \n {}".format(cfg))

    # 预热期设置
    cfg_trainer = cfg.trainer
    warmup_epochs = cfg_trainer.get("warmup_epochs", 0)
    logging.info("warmup_epochs: {}".format(warmup_epochs))
    print("warmup_epochs: ", warmup_epochs)

    # TODO: debug: save the real 3dmm:
    # for batch_idx, (_, _, speaker_3dmm_clip, _, _, _, _) in enumerate(tqdm(val_loader)):
    #     if batch_idx == 0:
    #         # print("batch size is: ", speaker_3dmm_clip.shape[0])
    #         speaker_3dmm_clip = speaker_3dmm_clip.detach().cpu().numpy()
    #         save_dir = os.path.join(cfg.trainer.out_dir, 'temp_save', 'exp_' + str(cfg.exp_num))
    #         os.makedirs(save_dir, exist_ok=True)
    #         np.save(os.path.join(save_dir, 'listener_3dmm.npy'), speaker_3dmm_clip)
    #         break

    for epoch in range(cfg.trainer.start_epoch, cfg.trainer.epochs):

        if epoch >= warmup_epochs:
            model.diffusion_decoder.warmup_ended = True
        else:
            model.diffusion_decoder.warmup_ended = False

        all_loss = (
            train(cfg, model, train_loader, optimizer, scheduler, criterion, epoch, writer, device))

        if cfg.loss.type in ["DiffusionSmoothLoss"]:
            train_loss, diff_prior_loss, diff_decoder_loss, temporal_loss, smooth_loss = all_loss
            logging.info(
                "Epoch: {} train_whole_loss: {:.5f} diff_prior_loss: {:.5f} diff_decoder_loss: {:.5f} "
                "temporal_loss: {:.5f} smooth_loss: {:.5f}"
                .format(epoch + 1, train_loss, diff_prior_loss, diff_decoder_loss, temporal_loss,  smooth_loss))
        elif cfg.loss.type in ["DiffusionSmoothPosLoss"]:
            train_loss, diff_prior_loss, diff_decoder_loss, temporal_loss, smooth_loss, pos_loss = all_loss
            logging.info(
                "Epoch: {} train_whole_loss: {:.5f} diff_prior_loss: {:.5f} diff_decoder_loss: {:.5f} "
                "temporal_loss: {:.5f} smooth_loss: {:.5f} pos_loss: {:.5f}"
                .format(epoch + 1, train_loss, diff_prior_loss, diff_decoder_loss, temporal_loss, smooth_loss, pos_loss))
        elif cfg.loss.type in ["DiffusionSmoothTransLoss"]:
            train_loss, diff_prior_loss, diff_decoder_loss, temporal_loss, smooth_loss, trans_loss = all_loss
            logging.info(
                "Epoch: {} train_whole_loss: {:.5f} diff_prior_loss: {:.5f} diff_decoder_loss: {:.5f} "
                "temporal_loss: {:.5f} smooth_loss: {:.5f} trans_loss: {:.5f}"
                .format(epoch + 1, train_loss, diff_prior_loss, diff_decoder_loss, temporal_loss, smooth_loss, trans_loss))
        else:
            train_loss, diff_prior_loss, diff_decoder_loss, temporal_loss = all_loss
            logging.info(
                "Epoch: {} train_whole_loss: {:.5f} diff_prior_loss: {:.5f} diff_decoder_loss: {:.5f} temporal_loss: {:.5f}"
                .format(epoch + 1, train_loss, diff_prior_loss, diff_decoder_loss, temporal_loss))

        if (epoch + 1) % cfg.trainer.val_period == 0:  # including the first epoch
            save_checkpoint(cfg, model, optimizer, (epoch+1), is_best=False)

        if (epoch + 1) % (cfg.trainer.val_period // 10) == 0:
            save_checkpoint(cfg, model, optimizer, (epoch+1), is_best=False, is_current=True)


if __name__ == '__main__':
    main(args=parse_arg())
