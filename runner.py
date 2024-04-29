import sys
import subprocess


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Need to supply command line arguments.')
        exit()
    gif_id = sys.argv[1]
    subprocess.run(['mkdir', f'gifs/{gif_id}-workspace'])
    subprocess.run(['cp', f'gifs/{gif_id}.gif', f'gifs/{gif_id}-workspace'])
    subprocess.run(['python', 'gif_to_rgb.py', f'gifs/{gif_id}-workspace/{gif_id}'])
    subprocess.run([f'(cd build && ./make)'])
    # subprocess.run(['make'])
    # subprocess.run([f'./gpu gifs/{gif_id}-workspace/{gif_id}'])
    # subprocess.run(['cd', '..'])

