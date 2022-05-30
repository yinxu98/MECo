import os
import random

percentage = 20

# folder_data = '../../data/opensarship'
folder_data = '../../data/gaofen4plane'
# folder_data = '../../data/rsscn'

ls_train = []
ls_val = []
classes = [d.name for d in os.scandir(folder_data) if d.is_dir()]

for class_name in classes:
    folder_class = os.path.join(folder_data, class_name)

    for _, _, fnames in sorted(os.walk(folder_class, followlinks=True)):
        random.shuffle(fnames)
        ls_train.extend([(class_name, fname)
                         for fname in fnames[:len(fnames) * percentage // 100]
                         ])
        ls_val.extend([(class_name, fname)
                       for fname in fnames[len(fnames) * percentage // 100:]])

with open(os.path.join(folder_data, f'train{percentage:d}.txt'), 'w') as fout:
    for class_name, fname in ls_train:
        fout.write(f'{class_name},{fname}\n')

with open(os.path.join(folder_data, f'val{percentage:d}.txt'), 'w') as fout:
    for class_name, fname in ls_val:
        fout.write(f'{class_name},{fname}\n')
