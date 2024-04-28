import ctypes
from PIL import Image
import numpy as np
import sys
np.set_printoptions(threshold=sys.maxsize)

def construct_gif(red_array, green_array, blue_array, path):
    num_frames, height, width = red_array.shape
    frames = []

    for i in range(num_frames):
        img = Image.new('RGB', (width, height))
        rgb_array = np.dstack((red_array[i], green_array[i], blue_array[i]))
        img = Image.fromarray(rgb_array.astype('uint8'))
        frames.append(img)

    frames[0].save(path, format='GIF', append_images=frames[1:], save_all=True, duration=100, loop=0)


if __name__ == '__main__':
    '''
    Reads individual (r, g, b) pixel values into a GIF file.
    '''
    gif_id = sys.argv[1]
    write_to = gif_id + '-modified.gif'
    with open(gif_id + '.red', 'r') as fp:
        dimensions = [int(dimen) for dimen in fp.readline().strip().split(',')]  # (num_frames, height, width)
    
    red_array = np.loadtxt(gif_id + '.red', skiprows=1)
    red_array = red_array.reshape(dimensions)

    green_array = np.loadtxt(gif_id + '.green', skiprows=1)
    green_array = green_array.reshape(dimensions)

    blue_array = np.loadtxt(gif_id + '.blue', skiprows=1)
    blue_array = blue_array.reshape(dimensions)

    construct_gif(red_array, green_array, blue_array, write_to)

