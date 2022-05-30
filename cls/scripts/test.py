import argparse
from datetime import datetime
from itertools import product
from os import path as osp

import mmcv
import torch
from addict import Dict
from cls.datasets import build_data_loader
from cls.models import build_model
from cls.tools import (AverageMeter, MetricsMeter, ProgressMeter, TimeMeter,
                       adjust_learning_rate, build_criterion_cls,
                       build_optimizer, calc_metrics, log, sanity_check)


def parse_args():
    parser = argparse.ArgumentParser(description='Test')
    parser.add_argument('config', help='test config file path')
    args = parser.parse_args()

    cfg = mmcv.Config.fromfile(args.config)

    cfg.work_dir = osp.join('../work_dirs',
                            osp.splitext(osp.basename(args.config))[0])
    mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))

    return cfg


def test(cfg):
    # Model
    model = build_model(cfg.model)

    # Dataset
    data_loader_train = build_data_loader(cfg.data, mode='train')
    data_loader_val = build_data_loader(cfg.data, mode='val')

    # Optimizer
    init_lr = cfg.optimizer.lr * cfg.data.batch_size.train / 512
    optimizer = build_optimizer(model, init_lr, cfg.optimizer)

    # Criterion
    criterion_cls = build_criterion_cls()

    # Test
    eta = TimeMeter(cfg.runner.epoch)
    metric = MetricsMeter(':.4f')

    for epoch in range(cfg.runner.epoch):
        losses = {'cls': AverageMeter('loss_cls', ':.4f')}
        progress = ProgressMeter(
            cfg.runner.epoch,
            [eta, losses['cls'], metric],
            prefix='Test: ',
        )

        eta.update_start(datetime.now())

        # train for one epoch
        model.train()

        adjust_learning_rate(optimizer, init_lr, epoch, cfg.runner.epoch)

        for img, gt in data_loader_train:
            img = img.cuda()
            gt = gt.cuda()

            # compute output
            pred = model(img)

            # compute loss
            loss_cls = criterion_cls(pred, gt)

            losses['cls'].update(loss_cls.item(), img.size(0))

            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss_cls.backward()
            optimizer.step()

        # validate for one epoch
        model.eval()

        confusion_matrix = torch.zeros(cfg.data.n_class, cfg.data.n_class)

        with torch.no_grad():
            for img, gt in data_loader_val:
                img = img.cuda()
                gt = gt.cuda()

                # compute output
                pred = model(img)

                # metrics
                pred = torch.max(pred, 1)[1].cpu()
                gt = gt.cpu()

                for i in range(len(gt)):
                    confusion_matrix[gt[i], pred[i]] += 1

        Pmi, F1 = calc_metrics(confusion_matrix)
        metric.update(dict(acc=Pmi, f1=F1))

        eta.update_end(datetime.now())

        # log
        txt = progress.str(epoch + 1)
        txt_conf_mat = ' conf_mat [' + ','.join(
            [str(int(cnt)) for cnt in confusion_matrix.view(-1)]) + ']'
        log(cfg.log_file, txt + txt_conf_mat + '\n')
        print(f'==> {txt}')

        # check sanity
        if epoch == 2:
            if cfg.test_mode == 'lin':
                sanity_check(model.backbone.state_dict(),
                             cfg.model.backbone.init_cfg.checkpoint)


def main():
    torch.backends.cudnn.benchmark = True

    cfg = parse_args()

    checkpoints = [
        osp.join(folder, index)
        for (folder,
             index) in product(cfg.checkpoints.folders, cfg.checkpoints.index)
    ]

    for checkpoint in checkpoints:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        cfg.log_file = osp.join(cfg.work_dir, f'{timestamp}.log')
        cfg.model.backbone.init_cfg = Dict(type=cfg.test_mode,
                                           checkpoint=checkpoint)
        cfg.dump(cfg.log_file)
        test(cfg)


if __name__ == '__main__':
    main()
