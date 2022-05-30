import argparse
from datetime import datetime
from os import path as osp

import mmcv
import torch
from cls.datasets import build_data_loader_single
from cls.models import build_model
from cls.tools import (AverageMeter, ProgressMeter, TimeMeter,
                       adjust_learning_rate, build_criterion_sim,
                       build_optimizer, log, save_checkpoint)


def parse_args():
    parser = argparse.ArgumentParser(description='Pretrain')
    parser.add_argument('config', help='pretrain config file path')
    args = parser.parse_args()

    cfg = mmcv.Config.fromfile(args.config)

    cfg.work_dir = osp.join('../work_dirs',
                            osp.splitext(osp.basename(args.config))[0])
    mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))

    return cfg


def train(cfg):
    # Model
    model = build_model(cfg.model)

    # Dataset
    data_loader = build_data_loader_single(cfg.data)

    # Optimizer
    init_lr = cfg.optimizer.lr * cfg.data.batch_size.pretrain / 512
    optimizer = build_optimizer(model, init_lr, cfg.optimizer)

    # Criterion
    criterion_sim = build_criterion_sim()

    # Pretrain
    eta = TimeMeter(cfg.runner.epoch)

    for epoch in range(cfg.runner.epoch):
        losses = {'sim': AverageMeter('loss_sim', ':.4f')}
        progress = ProgressMeter(
            cfg.runner.epoch,
            [eta, losses['sim']],
            prefix='Pretrain: ',
        )

        eta.update_start(datetime.now())

        # train for one epoch
        adjust_learning_rate(optimizer, init_lr, epoch, cfg.runner.epoch)

        for img in data_loader:
            img = img.cuda()

            # compute output
            p1, p2, r1, r2 = model(img)

            # compute loss
            loss_sim = -0.5 * (criterion_sim(p1, r2).mean() +
                               criterion_sim(p2, r1).mean())

            losses['sim'].update(loss_sim.item(), img.size(0))

            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss_sim.backward()
            optimizer.step()

        eta.update_end(datetime.now())

        # log and save
        txt = progress.str(epoch + 1)
        log(cfg.log_file, txt + '\n')
        print(f'==> {txt}')

        if (epoch + 1) % cfg.runner.save_interval == 0:
            save_checkpoint(
                model.backbone.state_dict(),
                dirname=cfg.work_dir,
                filename=f'{epoch + 1:0>4d}.pth',
            )


def main():
    torch.backends.cudnn.benchmark = True

    cfg = parse_args()

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    cfg.log_file = osp.join(cfg.work_dir, f'{timestamp}.log')
    cfg.dump(cfg.log_file)

    train(cfg)


if __name__ == '__main__':
    main()
