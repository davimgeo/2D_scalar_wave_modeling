import numpy as np
from numba import njit
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

class GenerateModel():
    def __init__(self, Nx, Nz, vp_interfaces, interfaces):
        self.Nx            = Nx
        self.Nz            = Nz
        self.vp_interfaces = vp_interfaces
        self.interfaces    = interfaces
        self.model         = self.set_model()
        
    def set_model(self):
        model = np.zeros((self.Nz, self.Nx))

        if len(self.interfaces) != len(self.vp_interfaces) - 1:
            raise Exception("interfaces must be a length smaller than velocity_interfaces")

        model[:self.interfaces[0], :] = self.vp_interfaces[0]
        for layer, velocity in enumerate(self.vp_interfaces[1:]):
            model[self.interfaces[layer]:, :] = velocity
        return model 
    
class GenerateGeometry():
    def __init__(self, spread, spacing, line_length):
        self.spread      = spread
        self.spacing     = spacing
        self.line_length = line_length
        self.geometry    = self.acquisition_geometry()

    def acquisition_geometry(self):
        total_shots = int(self.line_length / self.spacing) -  self.spread + 1
        total_nodes = total_shots + self.spread
        receivers = np.arange(total_nodes) * self.spacing

        if receivers[-1] > self.line_length: raise ValueError("Geometry too big for model")
        if total_nodes < 0: raise ValueError("Spacing or Spread too big for geometry!")

        initial_shot = (receivers[-1] - (total_shots - 1) * self.spacing) / 2
        sources = initial_shot + np.arange(total_shots) * self.spacing
        return sources, receivers
    
class GenerateWavelet():
    def __init__(self, Nt, dt, fmax):
        self.Nt             = Nt
        self.dt             = dt
        self.fmax           = fmax
        self.ricker_wavelet = self.set_ricker_wavelet()
        
    def set_ricker_wavelet(self):
        t0 = 2*np.pi/self.fmax 
        t = np.arange(self.Nt)*self.dt - t0

        arg = (np.pi*self.fmax*t) * (np.pi*self.fmax*t)

        return (1.0 - 2.0*arg)*np.exp(-arg)
    
class CheckParameters():
    def __init__(self, dx, dt, fmax, cmin, cmax):
        self.dx   = dx
        self.dt   = dt
        self.fmax = fmax
        self.cmin = cmin
        self.cmax = cmax
        self.check_parameters = self.check_parameters()

    def ideal_parameters(self):
        alpha = 5
        beta = 10
        ideal_dx = self.cmin / (alpha*self.fmax)
        ideal_dt = ideal_dx / (beta*self.cmax)
        return ideal_dx, ideal_dt
    
    def check_parameters(self):
        ideal_dx, ideal_dt = self.ideal_parameters()
        return self.dx <= ideal_dx and self.dt <= ideal_dt

class FiniteDifferences():
    def __init__(self, wavelet, Nx, Nz, Nt, dx, dz, dt, model, snap_bool, snap_num, rec, src, depthsrc, fmax):
        self.wavelet = wavelet
        self.Nx = Nx
        self.Nz = Nz
        self.Nt = Nt
        self.dx = dx
        self.dz = dz
        self.dt = dt
        self.fmax = fmax
        self.model = model
        self.snap_bool = snap_bool
        self.snap_num = snap_num
        self.rec = rec
        self.src = src
        self.depthsrc = depthsrc
        self.snapshots = []
        self.seismogram = np.zeros((Nt, len(rec)))
        self.Upas = np.zeros((Nz, Nx))
        self.Upre = np.zeros((Nz, Nx))
        self.Ufut = np.zeros((Nz, Nx))

    def finite_differences8E2T(self):
        srcxId = int(self.src[16])
        srczId = self.depthsrc

        recxId = self.rec
        reczId = np.zeros(len(self.rec)) + 20

        snap_ratio = int(self.Nt / self.snap_num)

        for n in range(self.Nt):
            self.register_seismogram(n, recxId, reczId)
            self.Upre[srczId, srcxId] += self.wavelet[n] / (self.dx * self.dx)
            laplacian_2D = self.get_laplacian_2D(self.Upre, self.Nx, self.Nz, self.dx, self.dz)
            self.Ufut = laplacian_2D * (self.dt * self.dt * self.model.model ** 2) + 2 * self.Upre - self.Upas
            self.get_snapshots(n, snap_ratio)
            self.Upas, self.Upre = self.Upre, self.Ufut

        return self.Ufut, self.snapshots, self.seismogram

    def register_seismogram(self, n, recx, recz):
        nrec = len(recx)
        for i in range(nrec):
            self.seismogram[n, i] = self.Upre[int(recz[i]), int(recx[i])]

    def get_snapshots(self, n, snap_ratio):
        if self.snap_bool and not n % snap_ratio:
            self.snapshots.append(self.Ufut)

    @staticmethod
    @njit(parallel=True)
    def get_laplacian_2D(U, Nx, Nz, dx, dz):
        d2u_dx2 = np.zeros((Nz, Nx))
        d2u_dz2 = np.zeros((Nz, Nx))
        for i in range(4, Nz - 4):
            for j in range(4, Nx - 4):
                d2u_dx2[i][j] = (-9*U[i-4,j] + 128*U[i-3,j] - 1008*U[i-2,j] + 8064*U[i-1,j]  \
                                - 14350*U[i,j] + 8064*U[i+1,j] -1008*U[i+2,j] + 128*U[i+3,j] \
                                -9*U[i+4,j]) / (5040*dx*dx)
                d2u_dz2[i][j] = (-9*U[i,j-4] + 128*U[i,j-3] - 1008*U[i,j-2] + 8064*U[i,j-1]  \
                                - 14350*U[i,j] + 8064*U[i,j+1] -1008*U[i,j+2] + 128*U[i,j+3] \
                                -9*U[i,j+4]) / (5040*dz*dz)
        return d2u_dx2 + d2u_dz2

class PlotGraph:
    def __init__(self, wave_propagation, start_index, snapshots):
        self.wave_propagation = wave_propagation
        self.start_index = start_index
        self.snapshots = snapshots
        self.geometry = GenerateGeometry(spread, spacing, line_length=Nx)

        self.fig, self.ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 5))

    def init(self):
        img = self.ax.imshow(self.wave_propagation[1][self.start_index], cmap='Greys', aspect='auto')
        self.ax.set_title("2D Wave Propagation", fontsize=18)
        return img,

    def update(self, i):
        self.ax.clear()
        img = self.ax.imshow(self.wave_propagation[1][i + self.start_index], cmap='Greys', aspect='auto')
        self.ax.set_title(f"snap_{i}", fontsize=18)
        return img,

    def plot_wave(self):
        if self.snapshots:
            self.ani = FuncAnimation(self.fig, self.update, init_func=self.init, frames=len(self.wave_propagation[1]) - self.start_index, interval=100, blit=True)
        else:
            self.ax.imshow(self.wave_propagation[0], cmap="Greys", aspect="auto")
            self.ax.plot(geometry.geometry[1], len(geometry.geometry[1])*[0], 'ko')
            self.ax.plot(geometry.geometry[0], len(geometry.geometry[0])*[0], 'bo')

        plt.figure()
        plt.title("Seismogram")
        plt.imshow(wave_propagation[2],aspect="auto",cmap="Greys")
        plt.yticks(np.linspace(0, Nt, 11), np.round(np.linspace(0, Nt, 11)*dt, 2))

if __name__ == "__main__":
    data = np.loadtxt("temp_par.txt", max_rows=1, dtype=int)
    decimal_data = np.genfromtxt('temp_par.txt', skip_header=3, max_rows=1, dtype=float)
    interfaces = np.genfromtxt('temp_par.txt', skip_header=2, max_rows=1, dtype=int).tolist()
    if isinstance(interfaces, int):
        interfaces = [interfaces]
    vp_interfaces = np.genfromtxt('temp_par.txt', skip_header=1, max_rows=1, dtype=int)

    Nx        = data[0]
    Nz        = data[1]
    Nx        = data[0]
    Nz        = data[1]
    Nt        = data[2]
    dx        = data[3]
    dz        = data[4]
    dt        = decimal_data
    fmax      = data[5]
    spread    = data[6]
    spacing   = data[7]
    depth_src = data[8]
    depth_rec = data[9]
    snapshots = True if data[10] == 1 else False
    snap_num  = data[11]

    model = GenerateModel(Nx, Nz, vp_interfaces, interfaces)
    geometry = GenerateGeometry(spread, spacing, line_length=Nx)
    wavelet = GenerateWavelet(Nt, dt, fmax)
    check_params = CheckParameters(dx, dt, fmax, min(vp_interfaces), max(vp_interfaces))
    fd = FiniteDifferences(wavelet.ricker_wavelet, Nx, Nz, Nt, dx, dz, dt, model, snapshots, \
                           snap_num, geometry.geometry[1], geometry.geometry[0], depth_src, fmax)
    wave_propagation = fd.finite_differences8E2T()

    start_index = int(0.2 * len(wave_propagation[1]))

    plot = PlotGraph(wave_propagation, start_index, snapshots)
    plot.plot_wave()

    plt.show()

    print(interfaces)