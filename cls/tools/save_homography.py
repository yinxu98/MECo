import random

import h5py
import numpy
from tqdm import tqdm


def find_coeffs(pa, pb):
    matrix = []
    for p1, p2 in zip(pa, pb):
        matrix.append(
            [p1[0], p1[1], 1, 0, 0, 0, -p2[0] * p1[0], -p2[0] * p1[1]])
        matrix.append(
            [0, 0, 0, p1[0], p1[1], 1, -p2[1] * p1[0], -p2[1] * p1[1]])

    A = numpy.matrix(matrix, dtype=numpy.float64)
    B = numpy.array(pb).reshape(8)

    res = numpy.dot(numpy.linalg.inv(A.T * A) * A.T, B)
    return numpy.array(res).reshape(8)


image_size = 64
all_coeffs = []

for _ in tqdm(range(500000)):
    width, height = (image_size, image_size)
    center = (image_size * 0.5 + 0.5, image_size * 0.5 + 0.5)

    shift = [float(random.randint(-10, 10)) for _ in range(8)]
    scale = random.uniform(0.8, 1.2)
    rotation = random.randint(0, 0)

    pts = [((0 - center[0]) * scale + center[0],
            (0 - center[1]) * scale + center[1]),
           ((width - center[0]) * scale + center[0],
            (0 - center[1]) * scale + center[1]),
           ((width - center[0]) * scale + center[0],
            (height - center[1]) * scale + center[1]),
           ((0 - center[0]) * scale + center[0],
            (height - center[1]) * scale + center[1])]
    pts = [pts[(ii + rotation) % 4] for ii in range(4)]
    pts = [(pts[ii][0] + shift[2 * ii], pts[ii][1] + shift[2 * ii + 1])
           for ii in range(4)]

    coeffs = find_coeffs(pts, [(0, 0), (width, 0), (width, height),
                               (0, height)])
    all_coeffs += [coeffs]

all_coeffs = numpy.asarray(all_coeffs, dtype=numpy.float64)
mean = numpy.mean(all_coeffs, axis=0)
std = numpy.std(all_coeffs, axis=0) + 1e-15
print('[' + ','.join([str(_) for _ in mean]) + ']')
print('[' + ','.join([str(_) for _ in std]) + ']')
with h5py.File('../datasets/homography.h5', 'w') as hf:
    hf.create_dataset('homography', data=all_coeffs)
