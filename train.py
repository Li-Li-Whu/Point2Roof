import os
import torch
import torch.nn as nn
import argparse
import datetime
import glob
import torch.distributed as dist
from dataset.data_utils import build_dataloader
from train_utils import train_model
from model.roofnet import RoofNet
from torch import optim
from utils import common_utils
from model import model_utils


def get_scheduler(optim, last_epoch):
    scheduler = torch.optim.lr_scheduler.StepLR(optim, 20, 0.5, last_epoch=last_epoch)
    return scheduler


def parse_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='../GithubDeepRoof', help='dataset path')
    parser.add_argument('--cfg_file', type=str, default='./model_cfg.yaml', help='model config for training')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size for training')
    parser.add_argument('--gpu', type=str, default='1', help='gpu for training')
    parser.add_argument('--extra_tag', type=str, default='pts6', help='extra tag for this experiment')
    parser.add_argument('--epochs', type=int, default=90, help='number of epochs to train for')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    args = parser.parse_args()
    cfg = common_utils.cfg_from_yaml_file(args.cfg_file)

    return args, cfg


def main():
    args, cfg = parse_config()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    extra_tag = args.extra_tag if args.extra_tag is not None \
            else 'model-%s' % datetime.datetime.now().strftime('%Y%m%d')
    output_dir = cfg.ROOT_DIR / 'output' / extra_tag
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = output_dir / 'ckpt'
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    log_file = output_dir / 'log.txt'
    logger = common_utils.create_logger(log_file)

    logger.info('**********************Start logging**********************')

    train_loader = build_dataloader(args.data_path, args.batch_size, cfg.DATA, training=True, logger=logger)

    net = RoofNet(cfg.MODEL)
    net.cuda()
    optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=1e-3)

    start_epoch = it = 0
    last_epoch = -1
    ckpt_list = glob.glob(str(ckpt_dir / '*checkpoint_epoch_*.pth'))
    if len(ckpt_list) > 0:
        ckpt_list.sort(key=os.path.getmtime)
        it, start_epoch = model_utils.load_params_with_optimizer(
            net, ckpt_list[-1], optimizer=optimizer, logger=logger
        )
        last_epoch = start_epoch + 1

    scheduler = get_scheduler(optimizer, last_epoch=last_epoch)

    net = net.train()
    logger.info('**********************Start training**********************')
    #logger.info(net)

    train_model(net, optimizer, train_loader, scheduler, it, start_epoch, args.epochs, ckpt_dir)


if __name__ == '__main__':
    main()
