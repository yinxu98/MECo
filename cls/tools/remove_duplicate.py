import os
from multiprocessing import Pool

import numpy as np
from PIL import Image, ImageChops
from tqdm import tqdm

folder_data = '/home/xuyin/data/gaofen4plane'


def run(ls_pair):
    ls_diff = []
    for pair in tqdm(ls_pair):
        im1, im2 = pair
        im1 = Image.open(im1)
        im2 = Image.open(im2)
        if im1.size == im2.size:
            continue
        diff = ImageChops.difference(im1, im2)
        diff = np.array(list(diff.getdata()))
        dmean = diff.mean()
        if dmean <= 12:
            dstd = diff.std()
            if dstd <= 12:
                ls_diff.append([pair[0], pair[1]])

    return ls_diff


if __name__ == '__main__':
    classes = [d.name for d in os.scandir(folder_data) if d.is_dir()]

    for class_name in classes:
        pool = Pool(30)
        result = []

        folder_class = os.path.join(folder_data, class_name)

        for _, _, fnames in sorted(os.walk(folder_class, followlinks=True)):
            fnames = [os.path.join(folder_class, fname) for fname in fnames]
            ls_pair = []
            for i in range(len(fnames)):
                for j in range(i):
                    ls_pair.append([fnames[i], fnames[j]])
            for i in range(0, len(ls_pair), 1000):
                result.append(
                    pool.apply_async(func=run, args=(ls_pair[i:i + 1000], )))

        pool.close()
        pool.join()

        ls_pair = []
        for res in result:
            ls_pair.extend(res.get())

        ls_sameset = []
        for im1, im2 in ls_pair:
            flag = False
            for sameset in ls_sameset:
                if im1 in sameset or im2 in sameset:
                    sameset.add(im1)
                    sameset.add(im2)
                    flag = True
                    break
            if not flag:
                ls_sameset.append(set((im1, im2)))

        for sameset in ls_sameset:
            sameset.pop()
            while len(sameset):
                file_img = sameset.pop()
                if os.path.exists(file_img):
                    os.remove(file_img)
