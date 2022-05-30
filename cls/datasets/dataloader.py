import os
import random
import time

import h5py
import numpy as np
import torch
from PIL import Image, ImageEnhance
from torch.utils.data.dataloader import DataLoader
from torchvision import datasets, transforms


class MyDataset(datasets.vision.VisionDataset):
    def __init__(self, cfg, mode):
        transform = transforms.Compose([
            transforms.Resize([cfg.image_size, cfg.image_size]),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=cfg.normalize.mean,
                std=cfg.normalize.std,
            ),
        ])

        super(MyDataset, self).__init__(cfg.root, transform=transform)

        self.mode = mode
        self.percentage = cfg.percentage

        self._make_dataset(cfg.root)

        if self.mode == 'pretrain':
            self._load_homography(cfg.homography)

    def _make_dataset(self, folder):
        classes = [d.name for d in os.scandir(folder) if d.is_dir()]
        classes.sort()
        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}

        if self.mode == 'pretrain':
            samples = []
            for class_name in classes:
                class_idx = class_to_idx[class_name]
                class_folder = os.path.join(folder, class_name)
                if not os.path.isdir(class_folder):
                    continue
                for root, _, fnames in sorted(
                        os.walk(class_folder, followlinks=True)):
                    for fname in sorted(fnames):
                        path = os.path.join(root, fname)
                        item = path, class_idx
                        samples.append(item)
        else:
            sample_file = os.path.join(folder,
                                       f'{self.mode}{self.percentage:d}.txt')
            with open(sample_file, 'r') as fin:
                ls_line = fin.readlines()
            ls_line = [line.strip('\n').split(',') for line in ls_line]
            samples = [(os.path.join(folder, class_name,
                                     fname), class_to_idx[class_name])
                       for (class_name, fname) in ls_line]

        self.samples, self.class_to_idx = samples, class_to_idx
        self.len = len(samples)

    def _load_homography(self, cfg):
        self.matrix_transform = transforms.Normalize(cfg.mean, cfg.std)

        with h5py.File(cfg.file, 'r') as hf:
            self.homography = hf['homography'][:]

    def _load_data(self, index):
        path, class_index = self.samples[index]
        with open(path, 'rb') as f:
            img = Image.open(f).convert('RGB')
        return img, class_index

    def _getitem_selfsup(self, index):
        rand_seed = index * self.len + time.time()
        random.seed(rand_seed)

        img0, _ = self._load_data(index % self.len)
        width, height = img0.size

        gt_brightness = random.uniform(0.5, 2)
        img1 = ImageEnhance.Brightness(img0).enhance(gt_brightness)

        gt_projective = self.homography[random.randint(0, 500000 - 1)]
        img2 = img1.transform((width, height),
                              Image.PERSPECTIVE,
                              gt_projective,
                              resample=False)

        gt_projective = torch.from_numpy(
            np.array(gt_projective, np.float64, copy=False)).view(8, 1, 1)
        gt_projective = self.matrix_transform(gt_projective).view(8)

        img = self.transform(img0)
        img_transformed = self.transform(img2)
        gt_param = torch.FloatTensor([gt_brightness, *gt_projective])

        return (img, img_transformed), gt_param

    def _getitem_sup(self, index):
        img, gt = self._load_data(index % self.len)
        img = self.transform(img)
        return img, gt

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return dict(
            pretrain=self._getitem_selfsup,
            train=self._getitem_sup,
            val=self._getitem_sup,
        ).get(self.mode)(index)


def build_data_loader(cfg, mode):
    dataset = MyDataset(cfg, mode)
    data_loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size[mode],
        shuffle=True,
        num_workers=cfg.workers,
        pin_memory=True,
        drop_last=False,
    )
    return data_loader


if __name__ == '__main__':
    from addict import Dict
    from torchvision.utils import make_grid, save_image

    data = Dict(
        root='../../data/gaofen4plane',
        normalize=dict(
            mean=[0.54601971, 0.54107395, 0.47337372],
            std=[0.20128778, 0.18832919, 0.19256465],
        ),
        homography=dict(
            file='./homography.h5',
            mean=[
                1.0690454325631715, 0.015830956719825373, -1.455018376998559,
                0.01567629779545185, 1.0692125828079984, -1.446146309283768,
                0.0004288037121747996, 0.00043649732070039576
            ],
            std=[
                0.40372048543771416, 0.19760619287460907, 10.208194354100314,
                0.19791173401645037, 0.40289598309784064, 10.178581044080417,
                0.004083203344468378, 0.004076959048045553
            ],
        ),
        workers=2,
        batch_size=dict(pretrain=25, train=512, val=2048),
    )

    # data_loader = build_data_loader(data, 'train')
    # input(data_loader.dataset.samples[:10])
    # data_loader = build_data_loader(data, 'val')
    # input(data_loader.dataset.samples[:10])

    data_loader = build_data_loader(data, 'pretrain')
    # input(data_loader.dataset.samples[:10])

    for images, gt_param in data_loader:
        img = make_grid(images[0], nrow=5)
        save_image(
            img,
            'img.jpg',
            normalize=True,
        )
        img_transformed = make_grid(images[1], nrow=5)
        save_image(
            img_transformed,
            'img_transformed.jpg',
            normalize=True,
        )
        input('save ok')
