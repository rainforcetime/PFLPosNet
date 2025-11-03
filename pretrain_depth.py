import logging
from functools import partial
import torch
from torch import optim
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm
import argparse

from dataset.dataset_depth import get_dataloader_depth_short
from dataset.dataset_embedder import get_dataloader
from utils import losses
from torch.utils.tensorboard import SummaryWriter
from utils.util import load_config, init_seed, get_logging_path, get_tensorboard_path, AverageMeter, \
    save_checkpoint_pretrain, compute_statistics, get_lr
import model as module_arch
import os


# python pretrain_depth.py --exp_num 1 --config ./config/depth_detector.yaml
def parse_arg():
    parser = argparse.ArgumentParser(description="PyTorch Training")
    # parser.add_argument("--exp_num", type=int, help="the number of the experiment.", required=True)
    parser.add_argument("--exp_num", type=int, help="the number of the experiment.", default=1)
    # parser.add_argument("--config", type=str, required=True)
    # parser.add_argument("--config", type=str, default="./config/latent_embedder.yaml")
    parser.add_argument("--config", type=str, default="./config/depth_detector.yaml")
    parser.add_argument("--gpu", type=int, help="the number of the gpu.", default=6)
    args = parser.parse_args()
    return args


def train(cfg, epoch, model, train_loader, optimizer, scheduler, criterion, writer, device):
    whole_losses = AverageMeter()

    # 使用 tqdm 包装 train_loader，并在描述中显示当前的 epoch
    progress_bar = tqdm(train_loader, desc=f'exp_{cfg.exp_num} Epoch {epoch + 1}/{cfg.trainer.epochs}', leave=True)

    model.train()
    for batch_idx, (video_clip, _3dmm_clip) in enumerate(progress_bar):
        video_clip, _3dmm_clip = video_clip.to(device), _3dmm_clip.to(device)

        output = model(x=video_clip, y=_3dmm_clip)
        loss_output = criterion(**output)

        loss = (
            loss_output["loss"]
        )
        # print(loss)

        iteration = epoch * len(train_loader) + batch_idx
        if writer is not None:
            writer.add_scalar("Train/whole_loss", loss.data.item(), iteration)

        whole_losses.update(loss.data.item(), video_clip.size(0))

        optimizer.zero_grad()
        loss.backward()
        if cfg.trainer.clip_grad:
            clip_grad_norm_(parameters=model.parameters(), max_norm=5, norm_type=2)
        optimizer.step()

    if scheduler is not None and (epoch + 1) >= 5:
        scheduler.step()

    # obtain the learning rate
    lr = get_lr(optimizer=optimizer)
    if writer is not None:
        writer.add_scalar("Train/lr", lr, epoch)

    return whole_losses.avg


def validate(cfg, epoch,  model, val_loader, criterion, writer, device):
    whole_losses = AverageMeter()
    model.eval()

    # 使用 tqdm 包装 train_loader，并在描述中显示当前的 epoch
    progress_bar = tqdm(val_loader, desc=f'exp_{cfg.exp_num} Epoch {epoch + 1}/{cfg.trainer.epochs}', leave=True)

    for batch_idx, (video_clip, _3dmm_clip) in enumerate(progress_bar):
        video_clip, _3dmm_clip = video_clip.to(device), _3dmm_clip.to(device)

        with torch.no_grad():
            output = model(x=video_clip, y=_3dmm_clip)
            loss_output = criterion(**output)

            loss = (
                loss_output["loss"]
            )

            iteration = epoch * len(val_loader) + batch_idx
            if writer is not None:
                writer.add_scalar("Val/whole_loss", loss.data.item(), iteration)

            whole_losses.update(loss.data.item(), video_clip.size(0))

    return whole_losses.avg


def main(args):
    gpu_id = args.gpu
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)  # Make only the first GPU visible
    os.environ['OMP_NUM_THREADS'] = '8'  # For debugging

    # load yaml config
    cfg = load_config(args=args, config_path=args.config)
    print("config: ", args.config)

    init_seed(seed=cfg.trainer.seed)  # seed initialization
    lowest_val_loss = 10000

    # logging
    logging_path = get_logging_path(cfg.trainer.log_dir, cfg.exp_num)
    os.makedirs(logging_path, exist_ok=True)
    logging.basicConfig(filename=logging_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')

    writer_path = get_tensorboard_path(cfg.trainer.tb_dir, cfg.exp_num)
    writer = SummaryWriter(writer_path)

    train_loader = get_dataloader_depth_short(cfg.dataset)
    val_loader = get_dataloader_depth_short(cfg.validation_dataset)

    # Set device ordinal if GPUs are available
    if torch.cuda.device_count() > 0:
        device = torch.device("cuda:0") # Adjust the device ordinal as needed
    else:
        device = torch.device("cpu")
    model = getattr(module_arch, cfg.model.type)(cfg.model.args)

    if cfg.trainer.resume is not None:
        checkpoint_path = cfg.trainer.resume
        print("Resume from {}".format(checkpoint_path))
        checkpoints = torch.load(checkpoint_path, map_location=torch.device('cpu'))
        state_dict = checkpoints['state_dict']
        model.load_state_dict(state_dict)
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

    criterion = partial(getattr(losses, cfg.loss.type))

    if cfg.optimizer.scheduler == "cosine_annealing":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_loader), eta_min=0,
                                                               last_epoch=-1)
    else:
        scheduler = None

    # log the config file
    logging.info("config: \n {}".format(cfg))

    for epoch in range(cfg.trainer.start_epoch, cfg.trainer.epochs):
        train_loss = (
            train(cfg, epoch, model, train_loader, optimizer, scheduler, criterion, writer, device)
        )

        logging.info(
            "Epoch: {} train_whole_loss: {:.5f}"
            .format(epoch + 1, train_loss))

        writer.add_scalar("Train/whole_loss", train_loss, epoch)

        if (epoch + 1) % cfg.trainer.val_period == 0:
            val_loss = validate(cfg, epoch, model, val_loader, criterion, writer, device)

            logging.info(
                "Epoch: {} val_whole_loss: {:.5f}"
                .format(epoch + 1, val_loss))

            writer.add_scalar("Val/whole_loss", val_loss, epoch)

            if val_loss < lowest_val_loss:
                lowest_val_loss = val_loss
                save_checkpoint_pretrain(cfg, model, optimizer, epoch, is_best=True)
            else:
                save_checkpoint_pretrain(cfg, model, optimizer, epoch, is_best=False)

    # Finally compute the statistics
    # cfg.trainer.resume = os.path.join(cfg.trainer.checkpoint_dir, 'best_checkpoint.pth')
    # compute_statistics(cfg, model, train_loader, device)

    writer.close()


if __name__ == '__main__':
    main(args=parse_arg())
