import os, sys
sys.path.append('..')
import numpy as np
import cv2
import multiprocessing
from scipy.sparse import diags
from scipy.sparse.linalg import cg
from scipy.sparse import csr_matrix

from options import config_parser
from utils.filter_utils import *
from utils.io_utils import *

#From https://github.com/jonbarron/planar_filter/blob/main/planar_filter.ipynb
def planar_filter(Z, filt, eps):
  # Solve for the plane at each pixel in `Z`, where the plane fit is computed
  # by using `filt` (a function that blurs something of the same size and shape
  # as `Z` by taking a linear non-negative combination of inputs) to weight
  # pixels in Z, and `eps` regularizes the output to be fronto-parallel.
  # Returns (Zx, Zy, Zz), which is a plane parameterization for each pixel:
  # the derivative wrt x and y, and the offset (which can itself be used as
  # "the" filtered output).

  # Note: This isn't the same code as in the paper. I flipped x and y to match
  # a more pythonic (x, y) convention, and I had to flip a sign on the output
  # slopes to make the unit tests pass(this may be a bug in the paper's math).
  # Also, I decided to not regularize the "offset" component of the plane fit,
  # which means that setting eps -> infinity gives the output (0, 0, filt(Z)).
  xy_shape = np.array(Z.shape[-2:])
  xy_scale = 2 / np.mean(xy_shape-1)  # Scaling the x, y coords to be in ~[0, 1]
  x, y = np.meshgrid(*[(np.arange(s) - (s-1)/2) * xy_scale for s in xy_shape], indexing='ij')
  [F1, Fx, Fy, Fz, Fxx, Fxy, Fxz, Fyy, Fyz] = [
    filt(t) for t in [
    np.ones_like(x), x, y, Z, x**2, x*y, x*Z, y**2, y*Z]]
  A11 = F1*x**2 - 2*x*Fx + Fxx + eps**2
  A22 = F1*y**2 - 2*y*Fy + Fyy + eps**2
  A12 = F1*y*x - x*Fy - y*Fx + Fxy
  A13 = F1*x - Fx
  A23 = F1*y - Fy
  A33 = F1# + eps**2
  b1 = Fz*x - Fxz
  b2 = Fz*y - Fyz
  b3 = Fz
  Zx, Zy, Zz = solve_image_ldl3(A11, A12, A13, A22, A23, A33, b1, b2, b3)
  return -Zx*xy_scale, -Zy*xy_scale, Zz


def get_valid_idx(valid, candidates):
    """Find which values are present in a list and where they are located"""
    locs = np.searchsorted(valid, candidates)
    # Handle edge case where the candidate is larger than all valid values
    locs = np.clip(locs, 0, len(valid) - 1)
    # Identify which values are actually present
    valid_idx = np.flatnonzero(valid[locs] == candidates)
    locs = locs[valid_idx]
    return valid_idx, locs

class BilateralGrid(object):
    def __init__(self, im, sigma_spatial=32, sigma_luma=8, sigma_chroma=8):
        im_yuv = rgb2yuv(im)
        # Compute 5-dimensional XYLUV bilateral-space coordinates
        Iy, Ix = np.mgrid[:im.shape[0], :im.shape[1]]
        x_coords = (Ix / sigma_spatial).astype(int)
        y_coords = (Iy / sigma_spatial).astype(int)
        luma_coords = (im_yuv[..., 0] /sigma_luma).astype(int)
        chroma_coords = (im_yuv[..., 1:] / sigma_chroma).astype(int)
        coords = np.dstack((x_coords, y_coords, luma_coords, chroma_coords))
        coords_flat = coords.reshape(-1, coords.shape[-1])
        self.npixels, self.dim = coords_flat.shape
        # Hacky "hash vector" for coordinates,
        # Requires all scaled coordinates be < MAX_VAL
        self.hash_vec = (MAX_VAL**np.arange(self.dim))
        # Construct S and B matrix
        self._compute_factorization(coords_flat)

    def _compute_factorization(self, coords_flat):
        # Hash each coordinate in grid to a unique value
        hashed_coords = self._hash_coords(coords_flat)
        unique_hashes, unique_idx, idx = \
            np.unique(hashed_coords, return_index=True, return_inverse=True)
        # Identify unique set of vertices
        unique_coords = coords_flat[unique_idx]
        self.nvertices = len(unique_coords)
        # Construct sparse splat matrix that maps from pixels to vertices
        self.S = csr_matrix((np.ones(self.npixels), (idx, np.arange(self.npixels))))
        # Construct sparse blur matrices.
        # Note that these represent [1 0 1] blurs, excluding the central element
        self.blurs = []
        for d in range(self.dim):
            blur = 0.0
            for offset in (-1, 1):
                offset_vec = np.zeros((1, self.dim))
                offset_vec[:, d] = offset
                neighbor_hash = self._hash_coords(unique_coords + offset_vec)
                valid_coord, idx = get_valid_idx(unique_hashes, neighbor_hash)
                blur = blur + csr_matrix((np.ones((len(valid_coord),)),
                                          (valid_coord, idx)),
                                         shape=(self.nvertices, self.nvertices))
            self.blurs.append(blur)

    def _hash_coords(self, coord):
        """Hacky function to turn a coordinate into a unique value"""
        return np.dot(coord.reshape(-1, self.dim), self.hash_vec)

    def splat(self, x):
        return self.S.dot(x)

    def slice(self, y):
        return self.S.T.dot(y)

    def blur(self, x):
        """Blur a bilateral-space vector with a 1 2 1 kernel in each dimension"""
        assert x.shape[0] == self.nvertices
        out = 2 * self.dim * x
        for blur in self.blurs:
            out += blur.dot(x)
        return out

    def filter(self, x):
        """Apply bilateral filter to an input x"""
        return self.slice(self.blur(self.splat(x))) /  \
               self.slice(self.blur(self.splat(np.ones_like(x))))


#From https://github.com/poolio/bilateral_solver
class BilateralSolver(object):
    def __init__(self, grid, params):
        self.grid = grid
        self.params = params
        self.Dn, self.Dm = bistochastize(grid)

    def solve(self, x, w):
        # Check that w is a vector or a nx1 matrix
        if w.ndim == 2:
            assert(w.shape[1] == 1)
        elif w.dim == 1:
            w = w.reshape(w.shape[0], 1)
        A_smooth = (self.Dm - self.Dn.dot(self.grid.blur(self.Dn)))
        w_splat = self.grid.splat(w)
        A_data = diags(w_splat[:,0], 0)
        A = self.params["lam"] * A_smooth + A_data
        xw = x * w
        b = self.grid.splat(xw)
        # Use simple Jacobi preconditioner
        A_diag = np.maximum(A.diagonal(), self.params["A_diag_min"])
        M = diags(1 / A_diag, 0)
        # Flat initialization
        y0 = self.grid.splat(xw) / (w_splat + 1e-10)
        yhat = np.empty_like(y0)
        for d in range(x.shape[-1]):
            yhat[..., d], info = cg(A, b[..., d], x0=y0[..., d], M=M, maxiter=self.params["cg_maxiter"], tol=self.params["cg_tol"])
        xhat = self.grid.slice(yhat)
        return xhat


def process(info):
    img, Z, W, save_path = info
    print('processing ' + save_path)
    grid_params = {
        'sigma_luma' : 8,
        'sigma_chroma': 8,
        'sigma_spatial': 8
    }

    bs_params = {
        'lam': 128, #The strength of the smoothness parameter
        'A_diag_min': 1e-5, # Clamp the diagonal of the A diagonal in the Jacobi preconditioner.
        'cg_tol': 1e-5, # The tolerance on the convergence in PCG
        'cg_maxiter': 25 # The number of PCG iterations
    }

    grid = BilateralGrid(img, **grid_params)

    def bf_solver(Z):
        t = Z.reshape(-1, 1).astype(np.double)
        c = W.reshape(-1, 1).astype(np.double)
        return BilateralSolver(grid, bs_params).solve(t, c).reshape(Z.shape)

    Zx, Zy, filter_depth = planar_filter(Z, bf_solver, 1)

    color_depth = visualize_depth(filter_depth)
    cv2.imwrite(save_path + '.png', color_depth)
    np.save(save_path + '.npy', filter_depth)
    

def main(args):
    print('Filtering begins !')
    image_list = load_img_list(args.datadir, load_test=False)
    pred_rgbs = load_rgbs_np(image_list, 
                    os.path.join(args.basedir, args.expname, 'nerf', 'results'),
                    is_png=True)
    pred_depths = load_depths(image_list, 
                    os.path.join(args.basedir, args.expname, 'nerf', 'results'))
    N, H, W, _ = pred_rgbs.shape
    gt_rgbs = load_rgbs_np(image_list, os.path.join(args.datadir, 'images'),
                           H=H, W=W)
    confidences = 1 - np.abs(pred_rgbs / 255.0 - gt_rgbs / 255.0).mean(-1)
    save_path = [os.path.join(args.basedir, args.expname, 'filter', img_name.split('.')[0]) 
                 for img_name in image_list]
    info_list = [(gt_rgbs[i], pred_depths[i], confidences[i], save_path[i]) 
                 for i in range(N)]
    p = multiprocessing.Pool(args.worker_num)
    p.map_async(process, info_list)
    p.close()
    p.join()

    
if __name__ == '__main__':
    parser = config_parser()
    args = parser.parse_args()
    main(args)