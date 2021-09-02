import sys
import torch
from torch.multiprocessing import set_start_method
set_start_method('spawn', force=True)
import time

from src import initialize, depth_priors, run_nerf, filter, evaluation
from utils.pose_utils import gen_poses
from options import config_parser
            
if __name__=='__main__':
    time = time.time()
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    parser = config_parser()
    args = parser.parse_args()
    print(args.expname)
    initialize.main(args)
    gen_poses(args.datadir)
    depth_priors.train(args)
    run_nerf.train(args)
    filter.main(args)
    
    if not args.demo:
        print('Evaluation begins !')
        evaluation.main(args)
    
    
    
    
    