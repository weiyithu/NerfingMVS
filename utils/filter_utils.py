import numpy as np
from scipy.sparse import diags


# From: https://en.wikipedia.org/wiki/YCbCr#JPEG_conversion
RGB_TO_YUV = np.array([
    [ 0.299,     0.587,     0.114],
    [-0.168736, -0.331264,  0.5],
    [ 0.5,      -0.418688, -0.081312]])
YUV_TO_RGB = np.array([
    [1.0,  0.0,      1.402],
    [1.0, -0.34414, -0.71414],
    [1.0,  1.772,    0.0]])
YUV_OFFSET = np.array([0, 128.0, 128.0]).reshape(1, 1, -1)

MAX_VAL = 255.0

def rgb2yuv(im):
    return (np.tensordot(im, RGB_TO_YUV, ([2], [1])) + YUV_OFFSET)

def yuv2rgb(im):
    return np.tensordot(im.astype(float) - YUV_OFFSET, YUV_TO_RGB, ([2], [1]))



def solve_image_ldl3(A11, A12, A13, A22, A23, A33, b1, b2, b3):
      # An unrolled LDL solver for a 3x3 symmetric linear system.
  d1 = A11
  L12 = A12/d1
  d2 = A22 - L12*A12
  L13 = A13/d1
  L23 = (A23 - L13*A12)/d2
  d3 = A33 - L13*A13 - L23*L23*d2 + 1e-10
  y1 = b1
  y2 = b2 - L12*y1
  y3 = b3 - L13*y1 - L23*y2
  x3 = y3/d3
  x2 = y2/d2 - L23*x3
  x1 = y1/d1 - L12*x2 - L13*x3
  return x1, x2, x3

# A simple linear blur filter. This can be whatever, provided it averages the
# input images by averaging its inputs with non-negative weights.
def blur(X, alpha):
  # Do an exponential decay filter on the outermost two dimensions of X.
  # Equivalent to convolving an image with a Laplacian blur.
  Y = X.copy()
  for i in range(Y.shape[-1]-1):
    Y[...,i+1] += alpha * Y[...,i]

  for i in range(Y.shape[-1]-1)[::-1]:
    Y[...,i] += alpha * Y[...,i+1]

  for i in range(Y.shape[-2]-1):
    Y[...,i+1,:] += alpha * Y[...,i,:]

  for i in range(Y.shape[-2]-1)[::-1]:
    Y[...,i,:] += alpha * Y[...,i+1,:]
  return Y

def bistochastize(grid, maxiter=10):
    """Compute diagonal matrices to bistochastize a bilateral grid"""
    m = grid.splat(np.ones(grid.npixels))
    n = np.ones(grid.nvertices)
    for i in range(maxiter):
        n = np.sqrt(n * m / (grid.blur(n) + 1e-10))
    # Correct m to satisfy the assumption of bistochastization regardless
    # of how many iterations have been run.
    m = n * grid.blur(n)
    Dm = diags(m, 0)
    Dn = diags(n, 0)
    return Dn, Dm
