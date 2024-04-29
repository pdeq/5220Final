#include "common.h"
#include <cuda.h>
#include <stdlib.h>

#define NUM_THREADS 256

// Put any static global variables here that you will use throughout the simulation.
int blks;

__device__ void apply_force_gpu(particle_t& particle, particle_t& neighbor) {
    double dx = neighbor.x - particle.x;
    double dy = neighbor.y - particle.y;
    double r2 = dx * dx + dy * dy;
    if (r2 > cutoff * cutoff)
        return;
    // r2 = fmax( r2, min_r*min_r );
    r2 = (r2 > min_r * min_r) ? r2 : min_r * min_r;
    double r = sqrt(r2);

    //
    //  very simple short-range repulsive force
    //
    double coef = (1 - cutoff / r) / r2 / mass;
    particle.ax += coef * dx;
    particle.ay += coef * dy;
}

__global__ void compute_forces_gpu(particle_t* particles, int num_parts) {
    // Get thread (particle) ID
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= num_parts)
        return;

    particles[tid].ax = particles[tid].ay = 0;
    for (int j = 0; j < num_parts; j++)
        apply_force_gpu(particles[tid], particles[j]);
}

__global__ void move_gpu(particle_t* particles, int num_parts, double size) {

    // Get thread (particle) ID
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= num_parts)
        return;

    particle_t* p = &particles[tid];
    //
    //  slightly simplified Velocity Verlet integration
    //  conserves energy better than explicit Euler method
    //
    p->vx += p->ax * dt;
    p->vy += p->ay * dt;
    p->x += p->vx * dt;
    p->y += p->vy * dt;

    //
    //  bounce from walls
    //
    while (p->x < 0 || p->x > size) {
        p->x = p->x < 0 ? -(p->x) : 2 * size - p->x;
        p->vx = -(p->vx);
    }
    while (p->y < 0 || p->y > size) {
        p->y = p->y < 0 ? -(p->y) : 2 * size - p->y;
        p->vy = -(p->vy);
    }
}

void init_simulation(particle_t* parts, int num_parts, double size) {
    // You can use this space to initialize data objects that you may need
    // This function will be called once before the algorithm begins
    // parts live in GPU memory
    // Do not do any particle simulation here

    blks = (num_parts + NUM_THREADS - 1) / NUM_THREADS;
}

void simulate_one_step(particle_t* parts, int num_parts, double size) {
    // parts live in GPU memory
    // Rewrite this function

    // Compute forces
    compute_forces_gpu<<<blks, NUM_THREADS>>>(parts, num_parts);

    // Move particles
    move_gpu<<<blks, NUM_THREADS>>>(parts, num_parts, size);
}

// Serial. Parallelize and put on GPU later
uint8_t *three_kernel(uint8_t *color_array, float *mask, int num_frames, int height, int width){
    uint8_t *masked_array = (uint8_t *) malloc(num_frames * height * width * sizeof(uint8_t));
    for (int i = 0; i < num_frames; ++i){
        for (int j = 1; j < height - 1; ++j){ // Avoid edges (at least for now)
            for (int k = 1; k < width - 1; ++k){ // Avoid edges
                int index = i * (width * height) + j * (width) + k;
                masked_array[index] = (uint8_t) (mask[4] * color_array[index]);
                masked_array[index] += (uint8_t) (mask[5] * color_array[index + 1] + mask[3] * color_array[index - 1]); // Horizontal neighbors
                masked_array[index] += (uint8_t) (mask[7] * color_array[index + width] + mask[1] * color_array[index - width]); // Vertical neighbors
                masked_array[index] += (uint8_t) (mask[8] * color_array[index + width + 1] + mask[2] * color_array[index - width + 1]); // Right diag
                masked_array[index] += (uint8_t) (mask[6] * color_array[index + width - 1] + mask[0] * color_array[index - width - 1]); // Left diag
            }
        }
    }
    return masked_array;
}

// Serial. Parallelize and put on GPU later
void tint_color(uint8_t *color_array, uint8_t color_val, float weight, int array_len){
    for (int i = 0; i < array_len; ++i){
        color_array[i] = (uint8_t) ((color_val + weight * color_array[i]) > 255 ? 255 : (color_val + weight * color_array[i]));

    }
}

// Serial. Parallelize and put on GPU later
void shade_color(uint8_t *color_array, uint8_t color_val, float weight, int array_len){
    for (int i = 0; i < array_len; ++i){
        color_array[i] = (uint8_t) ((color_val - weight * color_array[i]) < 0 ? 0 : (color_val - weight * color_array[i]));

    }
}

