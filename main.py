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

  acous = Acoustic(model, geom, cfg)

  acous.get_ricker()
  acous.set_damper()
  acous.fd()

  return acous

import matplotlib.pyplot as plt
if __name__ == "__main__":
  acous = main()

  acous.plot_snapshots()
  acous.plot_seismogram()

  plt.imshow(acous.transit_time)
  plt.show()


