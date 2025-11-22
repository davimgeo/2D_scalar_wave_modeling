import time

from src import Acoustic, Config

import numpy as np
import matplotlib.pyplot as plt

PATH = "config/parameters.toml"

def measure_runtime(func):
  def wrapper(*args, **kwargs):
    start = time.time()
    result = func(*args, **kwargs)
    end = time.time()
    print(f"Runtime: {round(end - start, 4)} seconds")
    return result

  return wrapper

import numpy as np

@measure_runtime
def main():
  cfg = Config(PATH).load()

  acous = Acoustic(cfg)

  acous.get_model()
  acous.set_boundary()
  acous.set_damper()
  acous.get_ricker()

  acous.fd()

  acous.plot_snapshots()
  acous.plot_seismogram()

  return acous

if __name__ == "__main__":
  acous = main()


