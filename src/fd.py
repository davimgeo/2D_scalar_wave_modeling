from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from numba import njit, prange

class Acoustic:
  def __init__(self, model: Model, geom: Geometry, c):
    self.model_obj = model
    self.geom = geom
    self.c = c

    self.model = model.model
    self.nx = model.nx
    self.nz = model.nz
    self.nb = model.nb
    self.nxx = model.nxx
    self.nzz = model.nzz

    self.recx = geom.recx
    self.recz = geom.recz
    self.srcxId = geom.srcxId
    self.srczId = geom.srczId
    self.nrec = geom.nrec

    self.nt = c.nt
    self.dt = c.dt
    self.save_seismogram = c.save_seismogram
    self.seismogram_output_path = c.seismogram_output_path
    self.seismogram = np.zeros((self.nt, self.nrec))

    self.dh = c.dh

    self.factor = c.factor
    self.damp2D = np.ones((self.nzz, self.nxx))

    self.fmax = c.fmax
    self.ricker = np.zeros(self.nt)

    self.upas = np.zeros((self.nzz, self.nxx))
    self.upre = np.zeros((self.nzz, self.nxx))
    self.ufut = np.zeros((self.nzz, self.nxx))

    self.perc = c.perc

    self.snap_path = c.snap_path
    self.snap_num = c.snap_num
    self.snap_bool = c.snap_bool
    self.snapshots = []

    self.transit_time = np.zeros((self.nzz, self.nxx))

  def get_ricker(self):
    t0 = 2*np.pi / self.fmax
    t = np.arange(self.nt) * self.dt - t0

    arg = (np.pi * self.fmax * t) ** 2

    self.ricker = (1.0 - 2.0*arg) * np.exp(-arg)

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
    d2u_dx2 = np.zeros((self.nzz, self.nxx))
    d2u_dz2 = np.zeros((self.nzz, self.nxx))

    snap_ratio = int(self.nt / self.snap_num)

    dh2 = self.dh * self.dh
    arg = self.dt * self.dt * self.model * self.model

    for i in range(len(self.srcxId)):
      for t in range(self.nt):
        ix = int(self.srcxId[i]) + self.nb
        iz = int(self.srczId[i]) + self.nb
        self.upre[iz, ix] += self.ricker[t] / dh2

        current_time = int(t * self.dt)
        laplacian2d(
            self.upre, d2u_dx2, d2u_dz2, 
            self.nzz, self.nxx, dh2, current_time,
            self.transit_time, self.upas
        )

        self.ufut = (d2u_dx2 + d2u_dz2 * arg) + 2 * self.upre - self.upas

        self.upas = self.upre * self.damp2D
        self.upre = self.ufut * self.damp2D

        for irec in range(self.nrec):
          rx = int(self.recx[irec]) + self.nb
          rz = int(self.recz[irec]) + self.nb
          self.seismogram[t, irec] = self.upre[rz, rx]

        if self.snap_bool:
          if not t % snap_ratio:
            self.snapshots.append(self.upre.copy())

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

  def plot_snapshots(self) -> None:
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

      ax.plot(self.recx, self.recz, 'bv')
      ax.plot(self.srcxId, self.srczId, 'r*')

      ims.append([model_frame, snap_frame])

    ani = animation.ArtistAnimation(
        fig, ims,
        interval=(self.nt / len(self.snapshots) + 1) * self.dt * 1e3,
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

  def plot_seismogram(self) -> None:
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

class Model:
  def __init__(self, c) -> None:
    self.nx = c.nx
    self.nz = c.nz
    self.nb = c.nb

    self.nxx = 2*self.nb + self.nx
    self.nzz = 2*self.nb + self.nz

    self.model = np.zeros((self.nz, self.nx))

    self.interfaces = c.interfaces
    self.value_interfaces = c.velocity_interfaces

  def get_model(self) -> None:
    if not len(self.interfaces):
      self.model[:, :] = self.value_interfaces[0]
    else:
      self.model[:self.interfaces[0], :] = self.value_interfaces[0]
      for layer, vel in enumerate(self.value_interfaces[1:]):
        self.model[self.interfaces[layer]:, :] = vel

  def set_boundary(self) -> None:
    model_ext = np.zeros((self.nzz, self.nxx))

    for j in range(self.nx):
      for i in range(self.nz):
        model_ext[i+self.nb, j+self.nb] = self.model[i, j]

    for j in range(self.nb, self.nx+self.nb):
      for i in range(self.nb):
        model_ext[i, j] = model_ext[self.nb, j]
        model_ext[self.nz+self.nb+i, j] = model_ext[self.nz+self.nb-1, j]

    for i in range(self.nzz):
      for j in range(self.nb):
        model_ext[i, j] = model_ext[i, self.nb]
        model_ext[i, self.nx+self.nb+j] = model_ext[i, self.nx+self.nb-1]

    self.model = model_ext

class Geometry:
  def __init__(self, c) -> None:
    self.c = c

    self.recx = []
    self.recz = []
    self.srcxId = []
    self.srczId = []

    self.nrec = 0

  def get_geometry(self) -> None:
    receivers = np.loadtxt(self.c.receivers, delimiter=',', skiprows=1)

    if receivers.ndim == 1:
      self.recx = np.array([receivers[1]])
      self.recz = np.array([receivers[2]])
    else:
      self.recx = receivers[:, 1]
      self.recz = receivers[:, 2]

    sources = np.loadtxt(self.c.sources, delimiter=',', skiprows=1)

    if sources.ndim == 1:
      self.srcxId = np.array([sources[1]])
      self.srczId = np.array([sources[2]])
    else:
      self.srcxId = sources[:, 1]
      self.srczId = sources[:, 2]

    self.nrec = len(self.recx)

@njit(parallel=True)
def laplacian2d(
    upre: np.ndarray, d2u_dx2: np.ndarray, d2u_dz2: np.ndarray, 
    nzz: int, nxx: int, dh2: float, current_time: int,
    transit_time: np.ndarray, upas
) -> None:
  inv_dh2 = 1.0 / (5040.0 * dh2)

  ref = upas.copy()
  for i in prange(4, nzz - 4):
    for j in range(4, nxx - 4):
      d2u_dx2[i, j] = (
          -9   * upre[i-4, j] + 128   * upre[i-3, j] - 1008 * upre[i-2, j] +
          8064 * upre[i-1, j] - 14350 * upre[i,   j] + 8064 * upre[i+1, j] -
          1008 * upre[i+2, j] + 128   * upre[i+3, j] - 9    * upre[i+4, j]
      ) * inv_dh2

      d2u_dz2[i, j] = (
          -9   * upre[i, j-4] + 128   * upre[i, j-3] - 1008 * upre[i, j-2] +
          8064 * upre[i, j-1] - 14350 * upre[i, j]   + 8064 * upre[i, j+1] -
          1008 * upre[i, j+2] + 128   * upre[i, j+3] - 9    * upre[i, j+4]
      ) * inv_dh2

      # Criterio da Amplitude Maxima - Andre Bulcao
      # if abs(u(Ω,t)) >= abs(ref(Ω)) then
      # ref(Ω) = u(Ω,t)
      # T(Ω) = t
      # endif

      if np.abs(upre[i][j]) >= np.abs(ref[i][j]):
        ref[i][j] = upre[i][j]
        transit_time[i][j] = current_time

