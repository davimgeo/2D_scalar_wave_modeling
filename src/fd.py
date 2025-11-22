import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from numba import njit

class Acoustic:
  def __init__(self, c):
    self.nx = c.nx
    self.nz = c.nz

    self.nb = c.nb
    self.factor = c.factor

    self.nxx = 2 * self.nb + self.nx
    self.nzz = 2 * self.nb + self.nz

    self.damp2D = np.ones((self.nzz, self.nxx))

    self.dh = c.dh

    self.model = np.zeros((self.nz, self.nx))

    self.interfaces = c.interfaces
    self.value_interfaces = c.velocity_interfaces

    self.nt = c.nt
    self.dt = c.dt

    self.fmax = c.fmax
    self.ricker = np.zeros(self.nt)

    self.upas = np.zeros((self.nzz, self.nxx))
    self.upre = np.zeros((self.nzz, self.nxx))
    self.ufut = np.zeros((self.nzz, self.nxx))

    self.recx = np.arange(0, self.nxx)
    self.recz = np.zeros(self.nxx) + 10
    self.nrec = len(self.recx)

    self.perc = c.perc

    self.save_seismogram = c.save_seismogram
    self.seismogram_output_path = c.seismogram_output_path
    self.seismogram = np.zeros((self.nt, self.nrec))
 
    self.snap_path = c.snap_path
    self.snap_num  = c.snap_num
    self.snap_bool = c.snap_bool

    self.snapshots = []

  def get_ricker(self):
    t0 = 2*np.pi / self.fmax
    t = np.arange(self.nt) * self.dt - t0

    arg = (np.pi * self.fmax * t) ** 2

    self.ricker = (1.0 - 2.0*arg) * np.exp(-arg)

  def get_model(self):
    if not len(self.interfaces):
      self.model[:, :] = self.value_interfaces[0]
    else:
      self.model[:self.interfaces[0], :] = self.value_interfaces[0]

      for layer, velocity in enumerate(self.value_interfaces[1:]):
        self.model[self.interfaces[layer]:, :] = velocity

  def set_boundary(self):
    model_ext  = np.zeros((self.nzz, self.nxx))
  
    for j in range(self.nx):
      for i in range(self.nz):
        model_ext[i + self.nb, j + self.nb] = self.model[i, j]

    for j in range(self.nb, self.nx + self.nb):
      for i in range(self.nb):
        # top
        model_ext[i, j] = model_ext[self.nb, j]

        # bottom
        model_ext[self.nz + self.nb + i, j] = model_ext[self.nz + self.nb - 1, j]

    for i in range(self.nzz):
      for j in range(self.nb):
        # left
        model_ext[i, j] = model_ext[i, self.nb]

        # right
        model_ext[i, self.nx + self.nb + j] = model_ext[i, self.nx + self.nb - 1]

    self.model = model_ext

  # Made by Paulo Bastos(https://github.com/phbastosa)
  def set_damper(self):
    damp1D = np.zeros(self.nb)

    for i in range(self.nb):   
        damp1D[i] = np.exp(-(self.factor*(self.nb - i))**2.0)

    for i in range(self.nzz):
        self.damp2D[i,:self.nb] *= damp1D
        self.damp2D[i,-self.nb:] *= damp1D[::-1]

    for j in range(self.nxx):
        self.damp2D[:self.nb,j] *= damp1D
        self.damp2D[-self.nb:,j] *= damp1D[::-1]   

  def fd(self):
    srcxId = self.nxx // 2
    srczId = self.nzz // 2

    snap_ratio = int(self.nt / self.snap_num)

    dh2 = self.dh * self.dh
    arg = self.dt * self.dt * self.model * self.model

    for n in range(self.nt):
      self.upre[srczId, srcxId] += self.ricker[n] / dh2

      laplacian_2D = get_laplacian_2D(self.upre, self.nxx, self.nzz, dh2)

      self.ufut = (laplacian_2D * arg) + 2*self.upre - self.upas

      self.upas = self.upre * self.damp2D
      self.upre = self.ufut * self.damp2D

      if self.snap_bool: 
        self.__get_snapshots(n, snap_ratio)

      for i in range(self.nrec):
        self.seismogram[n, i] = self.upre[int(self.recz[i]), int(self.recx[i])]

    if self.save_seismogram:
      (
        self.model
        .flatten('F')
        .astype("float32", order='F')
        .tofile(
          self.seismogram_output_path + 
          f"seismogram_nt{self.nt}_dt{self.dt}_nrec{self.nrec}.bin"
          ) 
      )

  def __get_snapshots(self, n, snap_ratio):
    if not n % snap_ratio:
      self.snapshots.append(self.upre.copy())

  def plot_snapshots(self):
    xloc = np.linspace(0, self.nx-1, 11, dtype=int)
    xlab = np.array(xloc * self.dh, dtype=int)

    zloc = np.linspace(0, self.nz-1, 7, dtype=int)
    zlab = np.array(zloc * self.dh, dtype=int)

    fig, ax = plt.subplots(figsize=(12, 5))

    ims = []
    for snap in self.snapshots:
      scale = 2.0 * np.std(snap)

      model_frame = ax.imshow(
          self.model[self.nb:self.nb+self.nz, self.nb:self.nb+self.nx],
          aspect="auto",
          cmap="jet",
          alpha=0.5
      )

      snap_frame = ax.imshow(
          snap[self.nb:self.nb+self.nz, self.nb:self.nb+self.nx],
          aspect="auto",
          cmap="Greys",
          vmin=-scale, vmax=scale,
          alpha=0.7
      )

      ims.append([model_frame, snap_frame])

    ani = animation.ArtistAnimation(
        fig, ims,
        interval=(self.nt / len(self.snapshots)) * self.dt * 1e3,
        blit=False,
        repeat_delay=0
    )

    ax.set_xticks(xloc)
    ax.set_xticklabels(xlab)
    ax.set_xlabel("Distance [m]", fontsize=15)

    ax.set_yticks(zloc)
    ax.set_yticklabels(zlab)
    ax.set_ylabel("Depth [m]", fontsize=15)

    fig.tight_layout()
    plt.show()

    return ani

  def plot_seismogram(self):
    scale_min = np.percentile(self.seismogram, 100 - self.perc)
    scale_max = np.percentile(self.seismogram, self.perc)

    fig, ax = plt.subplots(figsize=(10, 8))

    tloc = np.linspace(0, self.nt - 1, 11, dtype = int)
    tlab = np.around(tloc * self.dt, decimals = 1)

    xloc = np.linspace(0, self.nrec - 1, 9)
    xlab = np.array(10 * xloc, dtype = int)

    img = ax.imshow(
        self.seismogram, aspect = "auto", cmap="Greys", 
        vmin=scale_min, vmax=scale_max
    )

    cbar = fig.colorbar(img, ax = ax, extend = 'neither')
    cbar.minorticks_on()

    ax.set_yticks(tloc)
    ax.set_yticklabels(tlab)

    ax.set_xticks(xloc)
    ax.set_xticklabels(xlab)

    ax.set_title("Seismogram", fontsize = 18)
    ax.set_xlabel("Distance [m]", fontsize = 15)
    ax.set_ylabel("Two Way Time [s]", fontsize = 15)

    plt.tight_layout()
    plt.show()

@njit(parallel=True)
def get_laplacian_2D(U, nxx, nzz, dh2):
  d2u_dx2 = np.zeros((nzz, nxx))
  d2u_dz2 = np.zeros((nzz, nxx))

  for i in range(4, nzz - 4):
    for j in range(4, nxx - 4):
      d2u_dx2[i][j] = (
        -9*U[i-4, j] + 128*U[i-3, j] - 1008*U[i-2, j] + 8064*U[i-1, j]
        - 14350*U[i, j] + 8064*U[i+1, j] - 1008*U[i+2, j] + 128*U[i+3, j]
        - 9*U[i+4, j]
      ) / (5040 * dh2)

      d2u_dz2[i][j] = (
        -9*U[i, j-4] + 128*U[i, j-3] - 1008*U[i, j-2] + 8064*U[i, j-1]
        - 14350*U[i, j] + 8064*U[i, j+1] - 1008*U[i, j+2] + 128*U[i, j+3]
        - 9*U[i, j+4]
      ) / (5040 * dh2)

  return d2u_dx2 + d2u_dz2

