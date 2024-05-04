#include "common.h"
#include <cuda.h>
#include <stdio.h>
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
    int stride = blockDim.x * gridDim.x;
    for (int i = tid; i < array_len; i += stride) {
        d_color_array[i] = (int) (color_val + weight * d_color_array[i]);
    }
}

__global__ void d_mask3(int *d_color_array, int *d_masked_array, float *d_mask, int num_frames, int height, int width) {
    int array_len = num_frames * height * width;
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = tid; i < array_len; i += stride) {

        int frame_index = i % (width * height);
        int v_edge = frame_index % width;
        bool is_edge_pixel = frame_index < width || frame_index > (width * height) - width || v_edge == 0 || v_edge == width -1;

        if (is_edge_pixel) {
            d_masked_array[i] = d_color_array[i];
        } else {
            float acc = 0;
            acc += d_mask[4] * d_color_array[i];
            acc += d_mask[5] * d_color_array[i + 1] + d_mask[3] * d_color_array[i - 1]; // Horizontal neighbors
            acc += d_mask[7] * d_color_array[i + width] + d_mask[1] * d_color_array[i - width]; // Vertical neighbors
            acc += d_mask[8] * d_color_array[i + width + 1] + d_mask[2] * d_color_array[i - width + 1]; // Right diag
            acc += d_mask[6] * d_color_array[i + width - 1] + d_mask[0] * d_color_array[i - width - 1]; // Left diag
            d_masked_array[i] = (int) acc;
        }
    }
}


// Fast median of 9 values via http://ndevilla.free.fr/median/median/src/optmed.c
#define PIX_SORT(a,b) { if ((a)>(b)) PIX_SWAP((a),(b)); }
#define PIX_SWAP(a,b) { int temp=(a);(a)=(b);(b)=temp; }

__global__ void d_median3(int *d_color_array, int *d_masked_array, int num_frames, int height, int width){
    int array_len = num_frames * height * width;
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = tid; i < array_len; i += stride){
        int frame_index = i % (width * height);
        int v_edge = frame_index % width;
        bool is_edge_pixel = frame_index < width || frame_index > (width * height) - width || v_edge == 0 || v_edge == width - 1;

        if (is_edge_pixel){
            d_masked_array[i] = d_color_array[i];
        } else{
            // For serial code, it could be better to have a moving median,
            // but for GPU, I imagine it is better to just have each operation
            // be totally independent so each thread can just work on its own.
            int the_nine[] = {d_color_array[i], d_color_array[i + 1], d_color_array[i - 1],
                            d_color_array[i + width], d_color_array[i - width],
                            d_color_array[i + width + 1], d_color_array[i - width + 1],
                            d_color_array[i + width - 1], d_color_array[i - width - 1]};
            PIX_SORT(the_nine[1], the_nine[2]) ; PIX_SORT(the_nine[4], the_nine[5]) ;
            PIX_SORT(the_nine[7], the_nine[8]) ; PIX_SORT(the_nine[0], the_nine[1]) ;
            PIX_SORT(the_nine[3], the_nine[4]) ; PIX_SORT(the_nine[6], the_nine[7]) ;
            PIX_SORT(the_nine[1], the_nine[2]) ; PIX_SORT(the_nine[4], the_nine[5]) ;
            PIX_SORT(the_nine[7], the_nine[8]) ; PIX_SORT(the_nine[0], the_nine[3]) ;
            PIX_SORT(the_nine[5], the_nine[8]) ; PIX_SORT(the_nine[4], the_nine[7]) ;
            PIX_SORT(the_nine[3], the_nine[6]) ; PIX_SORT(the_nine[1], the_nine[4]) ;
            PIX_SORT(the_nine[2], the_nine[5]) ; PIX_SORT(the_nine[4], the_nine[7]) ;
            PIX_SORT(the_nine[4], the_nine[2]) ; PIX_SORT(the_nine[6], the_nine[4]) ;
            PIX_SORT(the_nine[4], the_nine[2]) ;
            d_masked_array[i] = the_nine[4];
        }
    }
}

__global__ void d_interpolate(int *d_color_array, int *d_interp_array, int num_frames, int height, int width) {
    // Loop ordering here
    int array_len = num_frames * height * width;
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = tid; i < array_len - width * height; i += stride) {
        int which_frame = i / (width * height);
        int frame_index = i % (width * height);
        
        // Port this pixel over to new array
        d_interp_array[(2 * which_frame) * width * height + frame_index] = 
            d_color_array[which_frame * width * height + frame_index];

        // 

        d_interp_array[(2 * which_frame + 1) * width * height + frame_index] = 0.5 * 
            (d_color_array[which_frame * width * height + frame_index] +
            d_color_array[(which_frame + 1) * width * height + frame_index]);
        
    }


}