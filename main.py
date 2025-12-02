import time

from src import *

PATH = "config/parameters.toml"

def measure_runtime(func):
  def wrapper(*args, **kwargs):
    start = time.time()
    result = func(*args, **kwargs)
    end = time.time()
    print(f"Runtime: {round(end - start, 4)} seconds")
    return result

  return wrapper

@measure_runtime
def main():
  cfg = Config(PATH).load()

  model = Model(cfg)
  model.get_model()
  model.set_boundary()

  geom = Geometry(cfg)
  geom.get_geometry()

  seis = Seismogram(geom, cfg)
  # load seismogram to remove direct wave
  seis.load()

  acous = Acoustic(model, geom, seis, cfg)

  acous.get_ricker()
  acous.set_damper()
  acous.fd()

  return acous, seis

import matplotlib.pyplot as plt
if __name__ == "__main__":
  acous, seis = main()

  acous.plot_snapshots()
  seis.plot(seis.seismogram)

  diff = seis.seismogram - seis.seismogram_load
  seis.plot(diff)

  #plt.imshow(acous.transit_time[acous.mdl.nb:acous.mdl.nb+acous.mdl.nz, acous.mdl.nb:acous.mdl.nb+acous.mdl.nx])
  #plt.show()


