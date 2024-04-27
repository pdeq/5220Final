#include <stdio.h>
#include <stdlib.h>
#include <gif_lib.h>

int main()
{
  int error = 0;
  char *file_name = "Animhorse.gif";
  GifFileType *gif_file = DGifOpenFileName(file_name, &error);
  DGifSlurp(gif_file);
  int num_frames = gif_file->ImageCount;
  int dimension = gif_file->SWidth * gif_file->SHeight;

  unsigned char *red_array = malloc(num_frames * dimension * sizeof(unsigned char));
  unsigned char *green_array = malloc(num_frames * dimension * sizeof(unsigned char));
  unsigned char *blue_array = malloc(num_frames * dimension * sizeof(unsigned char));

  int index = 0;
  for (int i = 0; i < num_frames; ++i)
  {
    unsigned char *this_frame = (gif_file->SavedImages[i]).RasterBits;
    for (int j = 0; j < dimension; ++j)
    {
      GifColorType color = gif_file->SColorMap->Colors[this_frame[j]];
      red_array[index] = color.Red;
      green_array[index] = color.Green;
      blue_array[index] = color.Blue;
      printf("Frame %d, Pixel %d: RGB(%u, %u, %u)\n", i, j, red_array[index], green_array[index], blue_array[index]);
      index++;
    }
  }

  free(red_array);
  free(green_array);
  free(blue_array);

  return 0;
}
