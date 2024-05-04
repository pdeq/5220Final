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
    duration = img.info['duration']

    for i in range(num_frames):
        img.seek(i)
        rgb = img.convert('RGB')
        r, g, b = rgb.split()
        red_array[i] = np.array(r)
        green_array[i] = np.array(g)
        blue_array[i] = np.array(b)

    return red_array, green_array, blue_array, duration


if __name__ == '__main__':
    '''
    Writes individual (r, g, b) pixel contents out to three files.
    '''
    gif_id = sys.argv[1]
    red_array, green_array, blue_array, duration = read_gif(gif_id + '.gif')
    num_frames, height, width = red_array.shape

    def write_gif(color, arr):
        with open(gif_id + color, 'w') as fp:
            fp.write(f'{num_frames}, {height}, {width}, {duration}\n')
            for frame in arr:
                np.savetxt(fp, frame, fmt='%d')
    
    write_gif('.red', red_array)
    write_gif('.green', green_array)
    write_gif('.blue', blue_array)

