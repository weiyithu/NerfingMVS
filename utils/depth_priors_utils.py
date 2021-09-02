import torch

def compute_depth_loss(depth_pred, depth_gt, mask_gt):
    loss_list = []
    for pred, gt, mask in zip(depth_pred, depth_gt, mask_gt):
        log_pred = torch.log(pred[mask])
        log_target = torch.log(gt[mask])
        alpha = (log_target - log_pred).sum()/mask.sum()
        log_diff = torch.abs((log_pred - log_target + alpha))
        d = 0.05*0.2*(log_diff.sum()/mask.sum())
        loss_list.append(d)

    return torch.stack(loss_list, 0).mean()