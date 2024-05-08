
int* three_kernel(int *color_array, float *mask, int num_frames, int height, int width);
void tint_color(int *color_array, int color_val, float weight, int array_len);
void shade_color(int *color_array, int color_val, float weight, int array_len);

__global__ void d_tint_color(int* d_color_array, int color_val, float weight, int array_len);
__global__ void d_mask3(int *d_color_array, int *d_masked_array, float *mask, int num_frames, int height, int width);
__global__ void d_median3(int *d_color_array, int *d_masked_array, int num_frames, int height, int width);
__global__ void d_interpolate(int *d_color_array, int *d_interp_array, int num_frames, int height, int width);
__global__ void d_gray_scale(int *d_color_array, int *d_red_array, int *d_green_array, int *d_blue_array, int num_frames, int height, int width);
__global__ void d_gauss(int *d_color, int num_frames, int height, int width, int *gauss);
__global__ void d_imgmul(int *d_color, int *d_right, int array_len);
__global__ void parallel_group(int *d_red, int *d_green, int *d_blue, float *means, int *assignments, int *assign_count, int width, int height, int num_frames, int k);
__global__ void parallel_means(int *d_red, int *d_green, int *d_blue, float *means, int *assignments, int *assign_count, int width, int height, int num_frames, int k);
__global__ void k_colors(int *d_red, int *d_green, int *d_blue, int *assignments, int *rs, int *gs, int *bs, int num_frames, int height, int width, int k);