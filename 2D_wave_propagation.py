import numpy as np
from numba import njit
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def set_model(Nx, Nz, dx, dz, velocity_interfaces, interfaces):
    model = np.zeros((Nz, Nx))

    if len(interfaces) != len(velocity_interfaces) - 1:
        raise Exception("interfaces and velocity_interfaces must be the same length!")

    model[:interfaces[0], :] = velocity_interfaces[0]
    for layer, velocity in enumerate(velocity_interfaces[1:]):
        model[interfaces[layer]:, :] = velocity
    return model

def set_wavelet(Nt, dt, fmax):
    t0 = 2*np.pi/fmax 
    t = np.arange(Nt)*dt - t0

    arg = (np.pi*fmax*t) * (np.pi*fmax*t)

    return (1.0 - 2.0*arg)*np.exp(-arg)

def check_parameters(dx, dt):
    ideal_dx, ideal_dt = ideal_parameters(min(vp_interfaces), max(vp_interfaces), fmax)
    if dx <= ideal_dx and dt <= ideal_dt:
        return True

def ideal_parameters(cmin, cmax, fmax):
    alpha = 5
    beta = 10
    ideal_dx = cmin / (alpha*fmax)
    ideal_dt = ideal_dx / (beta*cmax)
    return ideal_dx, ideal_dt

def get_snapshots(snap_bool, snapshots, snap_ratio, future_wave, n):
    if snap_bool:
        if not n % snap_ratio: snapshots.append(future_wave)
        return snapshots

def finite_differences(wavelet, Nx, Nz, Nt, dx, dz, dt, model, snap_bool, snap_num):

    if not check_parameters(dx, dt): raise ValueError("Current dx, dt are causing numerical dispersion!")

    srcxId = 208
    srczId = 10

    Upas = np.zeros((Nz, Nx))
    Upre = np.zeros((Nz, Nx))
    Ufut = np.zeros((Nz, Nx))

    snapshots = []
    snap_ratio = int(Nt / snap_num)

    for n in range(Nt):
        Upre[srczId,srcxId] += wavelet[n] / (dx*dx)

        laplacian_2D = get_laplacian_2D(Upre, Nx, Nz, dx, dz)

        Ufut = laplacian_2D*(dt*dt * model*model) + 2*Upre - Upas

        get_snapshots(snap_bool, snapshots, snap_ratio, Ufut, n)

        Upas, Upre = Upre, Ufut

    return Ufut, snapshots

@njit(parallel=True)
def get_laplacian_2D(U, Nx, Nz, dx, dz):
    d2u_dx2 = np.zeros((Nz, Nx))
    d2u_dz2 = np.zeros((Nz, Nx))

    for i in range(1, Nz-1):
        for j in range(1, Nx-1):
            d2u_dx2[i][j] = (U[i+1,j] - 2*U[i,j] + U[i-1,j]) / (dx*dx)
            #d2u_dx2[i][j] = (2*U[i-3,j] - 27*U[i-2,j] + 27*U[i-1,j] - 490*U[i,j] \
                             #+ 270*U[i+1,j] - 27*U[i+2,j] + 2*U[i+3,j]) / (180*dx*dx)
            d2u_dz2[i][j] = (U[i,j+1] - 2*U[i,j] + U[i,j-1]) / (dz*dz)
            #d2u_dz2[i][j] = (2*U[i,j-3] - 27*U[i,j-2] + 27*U[i,j-1] - 490*U[i,j] \
                             #+ 270*U[i,j+1] - 27*U[i,j+2] + 2*U[i,j+3]) / (180*dz*dz)
    return d2u_dx2 + d2u_dz2

def init():
    img = ax.imshow(wave_propagation[1][start_index], cmap='Greys', aspect='auto')
    ax.set_yticks(np.linspace(0, Nt, 11))
    ax.set_yticklabels(np.round(np.linspace(0, Nt, 11)*dt, 2))
    #ax.set_ylabel("TWT [s]", fontsize=15)
    ax.set_title("2D Wave Propagation", fontsize=18)
    return img,

def update(i):
    ax.clear()
    #scale = 1.5*np.std(wave_propagation[1][i])
    img = ax.imshow(wave_propagation[1][i+start_index], cmap='Greys', aspect='auto')
    ax.set_title(f"snap_{i}", fontsize=18)
    return img,

Nx = 416
Nz = 160
Nt = 5001

dx = 10
dz = 10
dt = 4e-4
fmax = 15

interfaces = [40, 100] 
vp_interfaces = [1500, 2000, 2500]

snapshots = True
snap_num = 100

model = set_model(Nx, Nz, dx, dz, vp_interfaces, interfaces)
ricker_wavelet = set_wavelet(Nt, dt, fmax)
wave_propagation = finite_differences(ricker_wavelet, Nx, Nz, Nt, dx, dz, dt, model, snapshots, snap_num)

start_index = int(0.2 * len(wave_propagation[1])) # x% of snapshot index
#print(ideal_parameters(min(vp_interfaces), max(vp_interfaces), fmax))

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10,5))

if snapshots:
    ani = FuncAnimation(fig, update, init_func=init, frames=len(wave_propagation[1]) - start_index, interval=100, blit=True)
else:
   ax.imshow(wave_propagation[0], cmap="Greys", aspect="auto") 

#ani.save('wave_propagation.mp4', writer='ffmpeg')

plt.show()