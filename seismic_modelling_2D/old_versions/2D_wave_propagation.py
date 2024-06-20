import numpy as np
from numba import njit
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def set_model(Nx, Nz, velocity_interfaces, interfaces):
    model = np.zeros((Nz, Nx))

    if len(interfaces) != len(velocity_interfaces) - 1:
        raise Exception("interfaces must be a length smaller than velocity_interfaces")

    model[:interfaces[0], :] = velocity_interfaces[0]
    for layer, velocity in enumerate(velocity_interfaces[1:]):
        model[interfaces[layer]:, :] = velocity
    return model

def acquisition_geometry(spread, spacing, line_length):
    total_shots = int(line_length / spacing) -  spread + 1
    total_nodes = total_shots + spread
    receivers = np.arange(total_nodes) * spacing

    if receivers[-1] > line_length: raise ValueError("Geometry too big for model")
    if total_nodes < 0: raise ValueError("Spacing or Spread too big for geometry!")

    initial_shot = (receivers[-1] - (total_shots - 1) * spacing) / 2
    sources = initial_shot + np.arange(total_shots) * spacing
    return sources, receivers

def set_wavelet(Nt, dt, fmax):
    t0 = 2*np.pi/fmax 
    t = np.arange(Nt)*dt - t0

    arg = (np.pi*fmax*t) * (np.pi*fmax*t)

    return (1.0 - 2.0*arg)*np.exp(-arg)

def check_parameters(dx, dt):
    ideal_dx, ideal_dt = ideal_parameters(min(vp_interfaces), max(vp_interfaces), fmax)
    return dx <= ideal_dx and dt <= ideal_dt

def ideal_parameters(cmin, cmax, fmax):
    alpha = 5
    beta = 10
    ideal_dx = cmin / (alpha*fmax)
    ideal_dt = ideal_dx / (beta*cmax)
    return ideal_dx, ideal_dt

def get_snapshots(snap_bool, snapshots, snap_ratio, future_wave, n):
    if snap_bool and not n % snap_ratio: snapshots.append(future_wave)
    return snapshots

def register_seismogram(n,recx,recz,seismogram,field):
    nrec = len(recx)
    for i in range(nrec):        
        seismogram[n,i] = field[int(recz[i]),int(recx[i])]

def finite_differences(wavelet, Nx, Nz, Nt, dx, dz, dt, model, snap_bool, snap_num, rec, src, depthsrc):

    if not check_parameters(dx, dt): raise ValueError("Current (dx, dt) are causing numerical dispersion!")

    srcxId = int(src[16])
    srczId = depthsrc

    recxId = rec
    reczId = np.zeros(len(rec)) + 20
    nrec = len(rec)
    seismogram = np.zeros((Nt, nrec))

    Upas = np.zeros((Nz, Nx))
    Upre = np.zeros((Nz, Nx))
    Ufut = np.zeros((Nz, Nx))

    snapshots = []
    snap_ratio = int(Nt / snap_num)

    for n in range(Nt):
        
        register_seismogram(n,recxId,reczId, seismogram, Upre)

        Upre[srczId,srcxId] += wavelet[n] / (dx*dx)

        laplacian_2D = get_laplacian_2D(Upre, Nx, Nz, dx, dz)

        Ufut = laplacian_2D*(dt*dt * model*model) + 2*Upre - Upas

        get_snapshots(snap_bool, snapshots, snap_ratio, Ufut, n)

        Upas, Upre = Upre, Ufut

    return Ufut, snapshots, seismogram

@njit(parallel=True)
def get_laplacian_2D(U, Nx, Nz, dx, dz):
    d2u_dx2 = np.zeros((Nz, Nx))
    d2u_dz2 = np.zeros((Nz, Nx))

    for i in range(4, Nz-4):
        for j in range(4, Nx-4):
            d2u_dx2[i][j] = (-9*U[i-4,j] + 128*U[i-3,j] - 1008*U[i-2,j] + 8064*U[i-1,j]  \
                            - 14350*U[i,j] + 8064*U[i+1,j] -1008*U[i+2,j] + 128*U[i+3,j] \
                            -9*U[i+4,j]) / (5040*dx*dx)
            d2u_dz2[i][j] = (-9*U[i,j-4] + 128*U[i,j-3] - 1008*U[i,j-2] + 8064*U[i,j-1]  \
                            - 14350*U[i,j] + 8064*U[i,j+1] -1008*U[i,j+2] + 128*U[i,j+3] \
                            -9*U[i,j+4]) / (5040*dz*dz)
    return d2u_dx2 + d2u_dz2

def init():
    img = ax.imshow(wave_propagation[1][start_index], cmap='Greys', aspect='auto')
    ax.set_title("2D Wave Propagation", fontsize=18)
    return img,

def update(i):
    ax.clear()
    #scale = 0.8*np.std(wave_propagation[1][i])
    img = ax.imshow(wave_propagation[1][i+start_index], cmap='Greys', aspect='auto')
    ax.set_title(f"snap_{i}", fontsize=18)
    return img,

Nx = 416
Nz = 160
Nt = 5001

dx = 10
dz = 10
dt = 4.0e-4
fmax = 10

spread = 52
spacing = 5

depth_src = 8
depth_rec = 0

interfaces = [40, 100] 
vp_interfaces = [1500, 2000, 2500]

snapshots = False
snap_num = 100

model = set_model(Nx, Nz, vp_interfaces, interfaces)
ricker_wavelet = set_wavelet(Nt, dt, fmax)
geometry = acquisition_geometry(spread, spacing, Nx)
wave_propagation = finite_differences(ricker_wavelet, Nx, Nz, Nt, dx, dz, dt, model, snapshots, snap_num, 
                                      geometry[1], geometry[0], depth_src)

start_index = int(0.2 * len(wave_propagation[1])) # x% of snapshot index
print(ideal_parameters(min(vp_interfaces), max(vp_interfaces), fmax))

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10,5))

if snapshots:
    ani = FuncAnimation(fig, update, init_func=init, frames=len(wave_propagation[1]) - start_index, interval=100, blit=True)
else:
   ax.imshow(wave_propagation[0], cmap="Greys", aspect="auto")
   ax.plot(geometry[1], len(geometry[1])*[0], 'ko')
   ax.plot(geometry[0], len(geometry[0])*[0], 'bo')

#ani.save('wave_propagation.mp4', writer='ffmpeg')

plt.figure()
plt.title("Seismogram")
plt.imshow(wave_propagation[2],aspect="auto",cmap="Greys")
plt.yticks(np.linspace(0, Nt, 11), np.round(np.linspace(0, Nt, 11)*dt, 2))

plt.show()  