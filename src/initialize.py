import os
from options import config_parser

def main(args):
    basedir = args.basedir
    expname = args.expname
    os.makedirs(os.path.join(basedir, expname), exist_ok=True)
    
    os.makedirs(os.path.join(basedir, expname, 'depth_priors', 'checkpoints'), exist_ok=True)
    os.makedirs(os.path.join(basedir, expname, 'depth_priors', 'results'), exist_ok=True)
    os.makedirs(os.path.join(basedir, expname, 'depth_priors', 'summary'), exist_ok=True)
    
    os.makedirs(os.path.join(basedir, expname, 'nerf', 'checkpoints'), exist_ok=True)
    os.makedirs(os.path.join(basedir, expname, 'nerf', 'results'), exist_ok=True)
    os.makedirs(os.path.join(basedir, expname, 'nerf', 'summary'), exist_ok=True)
    
    os.makedirs(os.path.join(basedir, expname, 'filter'), exist_ok=True)
    
    f = os.path.join(basedir, expname, 'args.txt')
    with open(f, 'w') as file:
        for arg in sorted(vars(args)):
            attr = getattr(args, arg)
            file.write('{} = {}\n'.format(arg, attr))
    if args.config is not None:
        f = os.path.join(basedir, expname, 'config.txt')
        with open(f, 'w') as file:
            file.write(open(args.config, 'r').read())
            
if __name__ == '__main__':
    parser = config_parser()
    args = parser.parse_args()
    main(args)