import os, sys
sys.path.append('..')
import torch

from utils.io_utils import *
from utils.evaluation_utils import *
from options import config_parser

def main(args):
    image_list = load_img_list(args.datadir, load_test=False)
    prior_path = os.path.join(args.basedir, args.expname, 'depth_priors', 'results')
    nerf_path = os.path.join(args.basedir, args.expname, 'nerf', 'results')
    filter_path = os.path.join(args.basedir, args.expname, 'filter')
    
    prior_depths = load_depths(image_list, prior_path)
    nerf_depths = load_depths(image_list, nerf_path)
    filter_depths = load_depths(image_list, filter_path)
    gt_depths, _ = load_gt_depths(image_list, args.datadir)
    
    print("prior depth evaluation:")
    depth_evaluation(gt_depths, prior_depths, savedir=prior_path)
    print("nerf depth evaluation:")
    depth_evaluation(gt_depths, nerf_depths, savedir=nerf_path)
    print("filter depth evaluation:")
    depth_evaluation(gt_depths, filter_depths, savedir=filter_path)
    
    image_list_all = load_img_list(args.datadir, load_test=True)
    image_list_test = list(set(image_list_all) - set(image_list))
    nerf_rgbs = load_rgbs_np(image_list_test, nerf_path, 
                             use_cv2=False, is_png=True)
    gt_rgbs = load_rgbs_np(image_list_test, 
                    os.path.join(args.datadir, 'images_{}'.format(args.factor)), 
                    use_cv2=False, is_png=True)
    print("nerf novel view synthesis evaluation:")
    with torch.no_grad():
        rgb_evaluation(gt_rgbs, nerf_rgbs, savedir=nerf_path)

if __name__ == '__main__':
    parser = config_parser()
    args = parser.parse_args()
    main(args)