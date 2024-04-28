import ctypes
from PIL import Image
import numpy as np
import sys
np.set_printoptions(threshold=sys.maxsize)

def read_gif(path):
  img = Image.open(path)

  num_frames = img.n_frames

  width, height = img.size

  red_array = np.zeros((num_frames, height, width), dtype='uint8')
  green_array = np.zeros((num_frames, height, width), dtype='uint8')
  blue_array = np.zeros((num_frames, height, width), dtype='uint8')

  for i in range(num_frames):
    img.seek(i)
    
    rgb = img.convert('RGB')
    
    r, g, b = rgb.split()
    
    red_array[i] = np.array(r)
    green_array[i] = np.array(g)
    blue_array[i] = np.array(b)
  return red_array, green_array, blue_array


def construct_gif(red_array, green_array, blue_array, path):
    num_frames, height, width = red_array.shape
    frames = []
    for i in range(num_frames):
        img = Image.new('RGB', (width, height))

        rgb_array = np.dstack((red_array[i], green_array[i], blue_array[i]))

        img = Image.fromarray(rgb_array.astype('uint8'))

        frames.append(img)

    frames[0].save(path, format='GIF', append_images=frames[1:], save_all=True, duration=100, loop=0)

a, b, c = read_gif("Animhorse.gif")

a[:] = 0

construct_gif(a, b, c, "out.gif")




gifproclib = ctypes.CDLL('./gifproclib.so')

gifproclib.lib_func()