import os, sys
sys.path.append('..')
import numpy as np
import torch
import cv2

from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm, trange
import pdb

from models.depth_priors.mannequin_challenge_model import MannequinChallengeModel
from options import config_parser
from utils.io_utils import *
from utils.depth_priors_utils import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def create_depth_model(args):
    """Instantiate depth model.
    """
    depth_model = MannequinChallengeModel()
    grad_vars = depth_model.parameters()
    # Create optimizer
    optimizer = torch.optim.Adam(params=grad_vars, lr=args.depth_lrate, betas=(0.9, 0.999))

    start = 0
    basedir = args.basedir
    expname = args.expname
    ckpt_path = os.path.join(basedir, expname, 'depth_priors', 'checkpoints')

    # Load checkpoints
    ckpts = [os.path.join(ckpt_path, f) for f in sorted(os.listdir(ckpt_path)) if 'tar' in f]

    if len(ckpts) > 0 and not args.no_reload:
        print('Found ckpts', ckpts)
        ckpt_path = ckpts[-1]
        print('Reloading from', ckpt_path)
        ckpt = torch.load(ckpt_path)

        start = ckpt['global_step']
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])

        decay_rate = 0.1
        decay_steps = args.depth_N_iters
        
        new_lrate = args.depth_lrate * (decay_rate ** (start / decay_steps))
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lrate

        depth_model.model.netG.load_state_dict(ckpt['netG_state_dict'])

    return depth_model, start, optimizer



def train(args):
    print('Depths prior training begins !')
    image_list = load_img_list(args.datadir)
    depth_model, global_step_depth, optimizer_depth = create_depth_model(args)

    # Summary writers
    save_dir = os.path.join(args.basedir, args.expname, 'depth_priors')
    writer = SummaryWriter(os.path.join(save_dir, 'summary'))

    
    images = load_rgbs(image_list, os.path.join(args.datadir, 'images'), 
                       args.depth_H, args.depth_W)
    images_train = images.clone()
    depths, masks = load_colmap(image_list, args.datadir, 
                                args.depth_H, args.depth_W)

    depths_train = torch.from_numpy(depths).to(device)
    depths_mask_train = torch.from_numpy(masks).to(device)

    N_rand_depth = args.depth_N_rand
    N_iters_depth = args.depth_N_iters
    
    i_batch = 0
    depth_model.train()
    start = global_step_depth + 1
    
    for i in trange(start, N_iters_depth):
        batch = images_train[i_batch:i_batch + N_rand_depth]
        depth_gt, mask_gt = depths_train[i_batch:i_batch + N_rand_depth], depths_mask_train[i_batch:i_batch + N_rand_depth]
        depth_pred = depth_model(batch)
        loss = compute_depth_loss(depth_pred, depth_gt, mask_gt)

        optimizer_depth.zero_grad()
        loss.backward()
        optimizer_depth.step()
        decay_rate = 0.1
        decay_steps = args.depth_N_iters
        new_lrate = args.depth_lrate * (decay_rate ** (i / decay_steps))
        for param_group in optimizer_depth.param_groups:
            param_group['lr'] = new_lrate
        i_batch += N_rand_depth
        
        if i_batch >= images_train.shape[0]:
            
            print("Shuffle depth data after an epoch!")
            rand_idx = torch.randperm(images_train.shape[0])
            images_train = images_train[rand_idx]
            depths_train = depths_train[rand_idx]
            depths_mask_train = depths_mask_train[rand_idx]
            i_batch = 0

        if i % args.depth_i_weights==0:
            path = os.path.join(save_dir, 'checkpoints', '{:06d}.tar'.format(i))
            torch.save({
                'global_step': i,
                'netG_state_dict': depth_model.model.netG.state_dict(),
                'optimizer_state_dict': optimizer_depth.state_dict(),
            }, path)
            print('Saved checkpoints at', path)
            
        if i%args.depth_i_print==0:
            tqdm.write(f"[TRAIN] Iter: {i} Loss: {loss.item()}")
        
        global_step_depth += 1
    print('depths prior training done!')
    
    with torch.no_grad():
        depth_model.eval()
        for i, image_name in enumerate(image_list):
            frame_id = image_name.split('.')[0]
            batch = images[i:i + 1]
            depth_pred = depth_model.forward(batch).cpu().numpy()
            depth_color = visualize_depth(depth_pred)
            cv2.imwrite(os.path.join(save_dir, 'results', '{}_depth.png'.format(frame_id)), depth_color)
            np.save(os.path.join(save_dir, 'results', '{}_depth.npy'.format(frame_id)), depth_pred)
    print('results have been saved in {}'.format(os.path.join(save_dir, 'results')))

if __name__=='__main__':
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    parser = config_parser()
    args = parser.parse_args()
    train(args)
