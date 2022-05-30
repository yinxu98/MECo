import os

from PIL import Image
from torch.utils.data.dataloader import DataLoader
from torchvision import datasets, transforms


class MyDatasetSingle(datasets.vision.VisionDataset):
    def __init__(self, cfg):
        transform = transforms.Compose([
            transforms.Resize([cfg.image_size, cfg.image_size]),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=cfg.normalize.mean,
                std=cfg.normalize.std,
            ),
        ])

        super(MyDatasetSingle, self).__init__(cfg.root, transform=transform)

        self._make_dataset(cfg.root)

    def _make_dataset(self, folder):
        classes = [d.name for d in os.scandir(folder) if d.is_dir()]
        classes.sort()
        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}

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

        self.samples, self.class_to_idx = samples, class_to_idx
        self.len = len(samples)

    def _load_data(self, index):
        path, class_index = self.samples[index]
        with open(path, 'rb') as f:
            img = Image.open(f).convert('RGB')
        return img, class_index

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        img, _ = self._load_data(index % self.len)
        img = self.transform(img)
        return img


def build_data_loader_single(cfg):
    dataset = MyDatasetSingle(cfg)
    data_loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size.pretrain,
        shuffle=True,
        num_workers=cfg.workers,
        pin_memory=True,
        drop_last=False,
    )
    return data_loader
