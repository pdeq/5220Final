
int* three_kernel(int *color_array, float *mask, int num_frames, int height, int width);
void tint_color(int *color_array, int color_val, float weight, int array_len);
void shade_color(int *color_array, int color_val, float weight, int array_len);

__global__ void d_tint_color(int* d_color_array, int color_val, float weight, int array_len);
__global__ void d_mask3(int *d_color_array, int *d_masked_array, float *mask, int num_frames, int height, int width);
__global__ void d_median3(int *d_color_array, int *d_masked_array, int num_frames, int height, int width);
