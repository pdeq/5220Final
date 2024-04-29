#include "common.h"
#include <cuda.h>
#include <stdlib.h>

#define NUM_THREADS 256

// Serial. Parallelize and put on GPU later
int *three_kernel(int *color_array, float *mask, int num_frames, int height, int width){
    int *masked_array = (int *) malloc(num_frames * height * width * sizeof(int));
    for (int i = 0; i < num_frames; ++i){
        for (int j = 1; j < height - 1; ++j){ // Avoid edges (at least for now)
            for (int k = 1; k < width - 1; ++k){ // Avoid edges
                int index = i * (width * height) + j * (width) + k;
                float acc = 0;
                acc += (mask[4] * color_array[index]);
                acc += (mask[5] * color_array[index + 1] + mask[3] * color_array[index - 1]); // Horizontal neighbors
                acc += (mask[7] * color_array[index + width] + mask[1] * color_array[index - width]); // Vertical neighbors
                acc += (mask[8] * color_array[index + width + 1] + mask[2] * color_array[index - width + 1]); // Right diag
                acc += (mask[6] * color_array[index + width - 1] + mask[0] * color_array[index - width - 1]); // Left diag
                masked_array[index] = (int) (acc);
            }
        }
    }
    return masked_array;
}

// Serial. Parallelize and put on GPU later
void tint_color(int *color_array, int color_val, float weight, int array_len){
    for (int i = 0; i < array_len; ++i){
        color_array[i] = (int) ((color_val + weight * color_array[i]) > 255 ? 255 : (color_val + weight * color_array[i]));

    }
}

// Serial. Parallelize and put on GPU later
void shade_color(int *color_array, int color_val, float weight, int array_len){
    for (int i = 0; i < array_len; ++i){
        color_array[i] = (int) ((color_val - weight * color_array[i]) < 0 ? 0 : (color_val - weight * color_array[i]));

    }
}

__global__ void d_tint_color(int* d_color_array, int color_val, float weight, int array_len) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= array_len) return;

    d_color_array[tid] = (int) ((color_val + weight * d_color_array[tid]) > 255 ? 255 : (color_val + weight * d_color_array[tid]));
}

