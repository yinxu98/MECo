import os

import numpy as np
from PIL import Image
from tqdm import tqdm

folder_root = '../../data/rsscn'

ls_folder = [d.name for d in os.scandir(folder_root) if d.is_dir()]

cnt_pixel = 0

R_channel = 0
G_channel = 0
B_channel = 0
for folder in ls_folder:
    path_folder = os.path.join(folder_root, folder)
    ls_image = os.listdir(path_folder)
    for image in tqdm(ls_image):
        path_image = os.path.join(path_folder, image)
        img = Image.open(path_image).convert('RGB')
        cnt_pixel += img.size[0] * img.size[1]
        img = np.array(img) / 255.0
        R_channel = R_channel + np.sum(img[:, :, 0])
        G_channel = G_channel + np.sum(img[:, :, 1])
        B_channel = B_channel + np.sum(img[:, :, 2])

R_mean = R_channel / cnt_pixel
G_mean = G_channel / cnt_pixel
B_mean = B_channel / cnt_pixel

R_channel = 0
G_channel = 0
B_channel = 0
for folder in ls_folder:
    path_folder = os.path.join(folder_root, folder)
    ls_image = os.listdir(path_folder)
    for image in tqdm(ls_image):
        path_image = os.path.join(path_folder, image)
        img = Image.open(path_image).convert('RGB')
        img = np.array(img) / 255.0
        R_channel = R_channel + np.sum((img[:, :, 0] - R_mean)**2)
        G_channel = G_channel + np.sum((img[:, :, 1] - G_mean)**2)
        B_channel = B_channel + np.sum((img[:, :, 2] - B_mean)**2)

R_std = np.sqrt(R_channel / cnt_pixel)
G_std = np.sqrt(G_channel / cnt_pixel)
B_std = np.sqrt(B_channel / cnt_pixel)

print(f'{R_mean},{G_mean},{B_mean}')
print(f'{R_std},{G_std},{B_std}')
