import argparse
import numpy as np
import os
import os.path as osp
import cv2
import struct

from .colmap_read_model import *

def read_array(path):
    with open(path, "rb") as fid:
        width, height, channels = np.genfromtxt(fid, delimiter="&", max_rows=1,
                                                usecols=(0, 1, 2), dtype=int)
        fid.seek(0)
        num_delimiter = 0
        byte = fid.read(1)
        while True:
            if byte == b"&":
                num_delimiter += 1
                if num_delimiter >= 3:
                    break
            byte = fid.read(1)
        array = np.fromfile(fid, np.float32)
    array = array.reshape((width, height, channels), order="F")
    return np.transpose(array, (1, 0, 2)).squeeze()

def load_point_vis(path, masks):
    with open(path, 'rb') as f:
        n = struct.unpack('<Q', f.read(8))[0]
        print('point number: {}'.format(n))
        for i in range(n):
            m = struct.unpack('<I', f.read(4))[0]
            for j in range(m):
                idx, u, v = struct.unpack('<III', f.read(4 * 3))
                masks[idx][v, u] = 1

def read_ply_mask(path):
    images_bin_path = os.path.join(os.path.dirname(path), 'sparse', 'images.bin')
    images = read_images_binary(images_bin_path)
    names = [dd[1].name for dd in images.items()]
    shapes = {}
    for name in names:
        depth_fname = os.path.join(os.path.dirname(path), 'stereo', 'depth_maps', name + '.geometric.bin')
        shapes[name] = read_array(depth_fname).shape

    ply_vis_path = path + '.vis'
    assert osp.exists(ply_vis_path)
    masks = [np.zeros(shapes[name], dtype=np.uint8) for name in names]
    load_point_vis(ply_vis_path, masks)
    return {name: mask for name, mask in zip(names, masks)}

