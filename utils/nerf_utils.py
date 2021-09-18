import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def align_scales(depth_priors, colmap_depths, colmap_masks, poses, sc, i_train, i_test):
    ratio_priors = []
    for i in range(depth_priors.shape[0]):
        ratio_priors.append(np.median(colmap_depths[i][colmap_masks[i]]) / np.median(depth_priors[i][colmap_masks[i]]))
    ratio_priors = np.stack(ratio_priors)
    ratio_priors = ratio_priors[:, np.newaxis, np.newaxis]
    
    if len(i_test) > 0:
        neighbor_idx = cal_neighbor_idx(poses, i_train, i_test)
        depth_priors_test = depth_priors[i_train][neighbor_idx]
        ratio_priors_test = ratio_priors[i_train][neighbor_idx]
        depth_priors = np.concatenate([depth_priors, depth_priors_test], axis=0)
        ratio_priors = np.concatenate([ratio_priors, ratio_priors_test], axis=0)

    depth_priors = depth_priors * sc * ratio_priors #align scales
    return depth_priors

def cal_depth_confidences(depths, T, K, i_train, topk=4):
    _, H, W = depths.shape
    view_num = len(i_train)
    invK = torch.inverse(K)
    batch_K = torch.unsqueeze(K, 0).repeat(view_num, 1, 1)
    batch_invK = torch.unsqueeze(invK, 0).repeat(depths.shape[0], 1, 1)
    T_train = T[i_train]
    invT = torch.inverse(T_train)
    pix_coords = calculate_coords(W, H)
    cam_points = BackprojectDepth(depths, batch_invK, pix_coords)
    depth_confidences = []

    for i in range(depths.shape[0]):
        cam_points_i = cam_points[i:i+1].repeat(view_num, 1, 1)
        T_i = torch.matmul(invT, T[i:i+1].repeat(view_num, 1, 1))
        pix_coords_ref = Project3D(cam_points_i, batch_K, T_i, H, W)
        depths_ = Project3D_depth(cam_points_i, batch_K, T_i, H, W)
        depths_proj = F.grid_sample(depths[i_train].unsqueeze(1), pix_coords_ref,
                        padding_mode="zeros").squeeze()
        error = torch.abs(depths_proj - depths_) / (depths_ + 1e-7)
        depth_confidence, _ = error.topk(k=topk, dim=0, largest=False)
        depth_confidence =  depth_confidence.mean(0).cpu().numpy()
        depth_confidences.append(depth_confidence)
    return np.stack(depth_confidences, 0)
    return np.stack(depth_confidences, 0)

def calculate_coords(W, H):
    meshgrid = np.meshgrid(range(W), range(H), indexing='xy')
    id_coords = np.stack(meshgrid, axis=0).astype(np.float32)
    id_coords = torch.from_numpy(id_coords)
    pix_coords = torch.stack(
            [id_coords[0].view(-1), id_coords[1].view(-1)], 0)
    ones = torch.ones(1, H * W)
    pix_coords = pix_coords.to(ones.device)
    pix_coords = torch.cat([pix_coords, ones], 0)
    return pix_coords

def BackprojectDepth(depth, invK, pix_coords):
    batch_size, H, W = depth.shape
    ones = torch.ones(batch_size, 1, H * W)
    cam_points = torch.matmul(invK[:, :3, :3], pix_coords)
    cam_points = depth.view(batch_size, 1, -1) * cam_points
    cam_points = torch.cat([cam_points, ones], 1)
    return cam_points

def Project3D(points, K, T, H, W, eps=1e-7):
    batch_size = points.shape[0]
    P = torch.matmul(K, T)[:, :3, :]

    cam_points = torch.matmul(P, points)

    pix_coords = cam_points[:, :2, :] / (cam_points[:, 2, :].unsqueeze(1) + eps)
    pix_coords = pix_coords.view(batch_size, 2, H, W)
    pix_coords = pix_coords.permute(0, 2, 3, 1)
    pix_coords[..., 0] /= W - 1
    pix_coords[..., 1] /= H - 1
    pix_coords = (pix_coords - 0.5) * 2
    return pix_coords

def Project3D_depth(points, K, T, H, W, eps=1e-7):
    batch_size = points.shape[0]
    P = torch.matmul(K, T)[:, :3, :]

    cam_points = torch.matmul(P, points)
    return cam_points[:, 2, :].view(batch_size, H, W)

def cal_neighbor_idx(poses, i_train, i_test):
    angles = []
    trans = []
    for i in range(poses.shape[0]):
        angles.append(vec_from_R(poses[i].copy()))
        trans.append(poses[i][:3,3].copy())
    angles, trans = np.stack(angles), np.stack(trans)
    angle_dis = angles[i_test][:, None] - angles[i_train][None, :]
    tran_dis = trans[i_test][:, None] - trans[i_train][None, :]
    angle_dis = (angle_dis ** 2).sum(-1)
    angle_sort = np.argsort(angle_dis, axis=1)
    tran_dis = (tran_dis ** 2).sum(-1)
    tran_sort = np.argsort(tran_dis, axis=1)
    x_range = np.arange(len(i_test))[:,None].repeat(len(i_train), axis=1)
    y_range = np.arange(len(i_train))[None].repeat(len(i_test), axis=0)
    angle_dis[x_range, angle_sort] = y_range
    tran_dis[x_range, tran_sort] = y_range
    final_score = 100*(angle_dis + tran_dis) + angle_dis
    neighbor_idx = np.argmin(final_score, axis=1)
    return neighbor_idx

def vec_from_R(rot):
    temp = (rot[:3,:3] - rot[:3,:3].transpose(1,0))/2
    angle_vec = np.stack([temp[2][1],-temp[2][0],temp[1][0]])
    angle = np.linalg.norm(angle_vec)
    axis = angle_vec/angle
    return np.arcsin(angle)*axis
