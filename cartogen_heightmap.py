# cartogen.py
import clvol
import pyopencl as cl
import ctypes as ct
import sys
import os

# {{{ cl setup
source_dir = os.path.dirname(sys.modules['cartogen'].__file__)
source_path = os.path.join(source_dir, 'heightmap.opencl')
source = open(source_path).read()
program = cl.Program(clvol.get_context(), source)
program.build(clvol.build_opts())

kernel_celestial_exposure = program.celestial_exposure
kernel_celestial_exposure.set_scalar_arg_dtypes((
  ct.c_int, ct.c_int,
  ct.c_float, ct.c_float,
  ct.c_float,
  ct.c_float,
  None,
  None))
# }}}

# {{{ celestial_exposure
def celestial_exposure(h, theta, phi, device_id = 0):
  h = h.resolve()
  tr = (0*h).resolve()
  q = clvol.get_queue(device_id)

  local_size = (16, 16, 1)
  global_size = clvol.global_size(h.shape, local_size, device_id)

  kernel_celestial_exposure(q, global_size, local_size,
      h.nx, h.ny,
      1, 1,
      theta,
      phi,
      h.as_buffer(device_id),
      tr.as_dirty_buffer(device_id))
  q.finish()

  h.free()
  tr.free()

  return tr
# }}}

# {{{ markov
def markov(nx, ny, eps = .001):
  ig = clvol.ImageGeom(nx, ny, 1)
  wx, wy, wz = clvol.fft3_freqs_even(ig.shape)
  v = clvol.div0(1, (2 + eps - clvol.cos(wx) - clvol.cos(wy))**2 )
  z = clvol.randn(ig.ones)

  U = clvol.FFT2(ig.shape)
  v = (U.t*(U*z * v)).real

  return ((v - v.min()) / (v.max() - v.min())).resolve()
# }}}

# {{{ islandifiy
def islandify(h, beta=1e2, niter=None):
    ig = clvol.ImageGeom(*h.shape)
    ix, iy, iz = clvol.fft3_indices(ig.shape)
    m = 2**(-.0001*((ix - ig.nx//2)**2 + (iy - ig.ny//2)**2))
    m = m.resolve()
    m = (m * (ix>=1)*(ix<ig.nx-1)*(iy>=1)*(iy<ig.ny-1)).resolve()

    # blast away edges
    omask = (ix>=1)*(ix<ig.nx-1)*(iy>=1)*(iy<ig.ny-1)
    h = (omask * h).resolve()

    R = clvol.Regularizer(ig.diff('2d8'), beta, ig.ones, 
            clvol.PotentialFunc('quad', (1,)))

    iter = niter
    if iter is None:
        iter = max(ig.nx, ig.ny)

    S = clvol.PCG(ig.eye, m, R, precon=clvol.Diag(omask.resolve()))

    return clvol.vmax(0, S(h, h, iter)).resolve()
# }}}

# {{{ watershed
def watershed(h, sea_level = 0):
  return ((h < sea_level) * sea_level).resolve()
# }}}

# {{{ draw
def draw(heightmap):
  h = heightmap.read()[:,:,0].T

  import pylab as pl
  pl.gray()
  f = pl.figure()
  ax = f.add_subplot(111)
  ax.imshow(h)

  ax.axis('off')
  return f, ax
# }}}

