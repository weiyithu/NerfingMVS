import os
import cv2
import numpy as np
import torch
import imageio
from torchvision import transforms

from .colmap_utils import *
import pdb

def load_img_list(datadir, load_test=False):
    with open(os.path.join(datadir, 'train.txt'), 'r') as f:
        lines = f.readlines()
        image_list = [line.strip() for line in lines]
        
    if load_test:
        with open(os.path.join(datadir, 'test.txt'), 'r') as f:
            lines = f.readlines()
            image_list += [line.strip() for line in lines]
    return image_list

def load_colmap(image_list, datadir, H=None, W=None):
    depths = []
    masks = []

    ply_path = os.path.join(datadir, 'dense', 'fused.ply')
    ply_masks = read_ply_mask(ply_path)

    for image_name in image_list:
        depth_path = os.path.join(datadir, 'dense/stereo/depth_maps', image_name + '.geometric.bin')
        depth = read_array(depth_path)
        mask = ply_masks[image_name]
        if H is not None:
            depth_resize = cv2.resize(depth, (W, H))
            mask_resize = cv2.resize(mask, (W, H))
        depths.append(depth_resize)
        masks.append(mask_resize > 0.5)

    return np.stack(depths), np.stack(masks)

def load_gt_depths(image_list, datadir, H=None, W=None):
    depths = []
    masks = []

    for image_name in image_list:
        frame_id = image_name.split('.')[0]
        depth_path = os.path.join(datadir, 'depth', '{}.png'.format(frame_id))
        depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        depth = depth.astype(np.float32) / 1000
        
        if H is not None:
            mask = (depth > 0).astype(np.uint8)
            depth_resize = cv2.resize(depth, (W, H), interpolation=cv2.INTER_NEAREST)
            mask_resize = cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST)
            depths.append(depth_resize)
            masks.append(mask_resize > 0.5)
        else:
            depths.append(depth)
            masks.append(depth > 0)

    return np.stack(depths), np.stack(masks)

def load_depths(image_list, datadir, H=None, W=None):
    depths = []

    for image_name in image_list:
        frame_id = image_name.split('.')[0]
        depth_path = os.path.join(datadir, '{}_depth.npy'.format(frame_id))
        if not os.path.exists(depth_path):
            depth_path = os.path.join(datadir, '{}.npy'.format(frame_id))
        depth = np.load(depth_path)
        
        if H is not None:
            depth_resize = cv2.resize(depth, (W, H))
            depths.append(depth_resize)
        else:
            depths.append(depth)

    return np.stack(depths)

def pil_loader(path):
    from PIL import Image
    # open path as file to avoid ResourceWarning
    # (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')

def load_rgbs(image_list, datadir, H=None, W=None, is_png=False):
    from PIL import Image
    to_tensor = transforms.ToTensor()
    resize = transforms.Resize((H, W), interpolation=Image.ANTIALIAS)
    rgbs = []

    for image_name in image_list:
        if is_png:
            image_name = image_name.replace('.jpg', '.png')
        rgb_path = os.path.join(datadir, image_name)
        rgb = pil_loader(rgb_path)
        if H is not None:
            rgb = resize(rgb)

        rgbs.append(to_tensor(rgb))

    return torch.stack(rgbs)

def load_rgbs_np(image_list, datadir, H=None, W=None, is_png=False, use_cv2=True):
    rgbs = []

    for image_name in image_list:
        if is_png:
            image_name = image_name.replace('.jpg', '.png')
        rgb_path = os.path.join(datadir, image_name)
        if use_cv2:
            rgb = cv2.imread(rgb_path)
        else:
            rgb = imageio.imread(rgb_path)[..., :3] / 255.0
        
        if H is not None:
            if use_cv2:    
                rgb = cv2.resize(rgb, (W, H))
            else:
                rgb = resize(rgb, (W, H))

        rgbs.append(rgb)

    return np.stack(rgbs)

def visualize_depth(depth, mask=None, depth_min=None, depth_max=None, direct=False):
    """Visualize the depth map with colormap.
       Rescales the values so that depth_min and depth_max map to 0 and 1,
       respectively.
    """
    if not direct:
        depth = 1.0 / (depth + 1e-6)
    invalid_mask = np.logical_or(np.isnan(depth), np.logical_not(np.isfinite(depth)))
    if mask is not None:
        invalid_mask += np.logical_not(mask)
    if depth_min is None:
        depth_min = np.percentile(depth[np.logical_not(invalid_mask)], 5)
    if depth_max is None:
        depth_max = np.percentile(depth[np.logical_not(invalid_mask)], 95)
    depth[depth < depth_min] = depth_min
    depth[depth > depth_max] = depth_max
    depth[invalid_mask] = depth_max

    depth_scaled = (depth - depth_min) / (depth_max - depth_min)
    depth_scaled_uint8 = np.uint8(depth_scaled * 255)
    depth_color = cv2.applyColorMap(depth_scaled_uint8, cv2.COLORMAP_MAGMA)
    depth_color[invalid_mask, :] = 0

    return depth_color