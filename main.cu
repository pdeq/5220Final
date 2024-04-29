#include "common.h"
#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include <cmath>
#include <cstring>
#include <cuda.h>
#include <fstream>
#include <iostream>
#include <random>
#include <vector>
#include <sstream>

// =================
// Helper Functions
// =================

#define IS_PETER true
std::string MY_PATH;
std::string GIF_ID;

// I/O routines
void save(std::ofstream& fsave, particle_t* parts, int num_parts, double size) {
    static bool first = true;

    if (first) {
        fsave << num_parts << " " << size << std::endl;
        first = false;
    }

    for (int i = 0; i < num_parts; ++i) {
        fsave << parts[i].x << " " << parts[i].y << std::endl;
    }

    fsave << std::endl;
}

// Particle Initialization
void init_particles(particle_t* parts, int num_parts, double size, int part_seed) {
    std::random_device rd;
    std::mt19937 gen(part_seed ? part_seed : rd());

    int sx = (int)ceil(sqrt((double)num_parts));
    int sy = (num_parts + sx - 1) / sx;

    std::vector<int> shuffle(num_parts);
    for (int i = 0; i < shuffle.size(); ++i) {
        shuffle[i] = i;
    }

    for (int i = 0; i < num_parts; ++i) {
        // Make sure particles are not spatially sorted
        std::uniform_int_distribution<int> rand_int(0, num_parts - i - 1);
        int j = rand_int(gen);
        int k = shuffle[j];
        shuffle[j] = shuffle[num_parts - i - 1];

        // Distribute particles evenly to ensure proper spacing
        parts[i].x = size * (1. + (k % sx)) / (1 + sx);
        parts[i].y = size * (1. + (k / sx)) / (1 + sy);

        // Assign random velocities within a bound
        std::uniform_real_distribution<float> rand_real(-1.0, 1.0);
        parts[i].vx = rand_real(gen);
        parts[i].vy = rand_real(gen);
    }
}

// Command Line Option Processing
int find_arg_idx(int argc, char** argv, const char* option) {
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], option) == 0) {
            return i;
        }
    }
    return -1;
}

int find_int_arg(int argc, char** argv, const char* option, int default_value) {
    int iplace = find_arg_idx(argc, argv, option);

    if (iplace >= 0 && iplace < argc - 1) {
        return std::stoi(argv[iplace + 1]);
    }

    return default_value;
}

char* find_string_option(int argc, char** argv, const char* option, char* default_value) {
    int iplace = find_arg_idx(argc, argv, option);

    if (iplace >= 0 && iplace < argc - 1) {
        return argv[iplace + 1];
    }

    return default_value;
}

// ==============
// Main Function
// ==============

// int main(int argc, char** argv) {
//     // Parse Args
//     if (find_arg_idx(argc, argv, "-h") >= 0) {
//         std::cout << "Options:" << std::endl;
//         std::cout << "-h: see this help" << std::endl;
//         std::cout << "-n <int>: set number of particles" << std::endl;
//         std::cout << "-o <filename>: set the output file name" << std::endl;
//         std::cout << "-s <int>: set particle initialization seed" << std::endl;
//         return 0;
//     }

//     // Open Output File
//     char* savename = find_string_option(argc, argv, "-o", nullptr);
//     std::ofstream fsave(savename);

//     // Initialize Particles
//     int num_parts = find_int_arg(argc, argv, "-n", 1000);
//     int part_seed = find_int_arg(argc, argv, "-s", 0);
//     double size = sqrt(density * num_parts);

//     particle_t* parts = new particle_t[num_parts];

//     init_particles(parts, num_parts, size, part_seed);

//     particle_t* parts_gpu;
//     cudaMalloc((void**)&parts_gpu, num_parts * sizeof(particle_t));
//     cudaMemcpy(parts_gpu, parts, num_parts * sizeof(particle_t), cudaMemcpyHostToDevice);

//     // Algorithm
//     auto start_time = std::chrono::steady_clock::now();

//     init_simulation(parts_gpu, num_parts, size);

//     for (int step = 0; step < nsteps; ++step) {
//         simulate_one_step(parts_gpu, num_parts, size);
//         cudaDeviceSynchronize();

//         // Save state if necessary
//         // if (fsave.good() && (step % savefreq) == 0) {
//         //     cudaMemcpy(parts, parts_gpu, num_parts * sizeof(particle_t), cudaMemcpyDeviceToHost);
//         //     save(fsave, parts, num_parts, size);
//         // }
//     }

//     cudaDeviceSynchronize();
//     auto end_time = std::chrono::steady_clock::now();

//     std::chrono::duration<double> diff = end_time - start_time;
//     double seconds = diff.count();

//     // Finalize
//     std::cout << "Simulation Time = " << seconds << " seconds for " << num_parts << " particles.\n";
//     fsave.close();
//     cudaFree(parts_gpu);
//     delete[] parts;
// }


void output_array(uint8_t* arr, std::string color, int num_frames, int height, int width) {
    std::ofstream file(MY_PATH + GIF_ID + "-modified." + color);
    if (!file.is_open()) std::cerr << "Cannot open output file." << std::endl;

    // Add dimensions to top of file
    std::ostringstream dimension_buffer;
    dimension_buffer << num_frames << ", " << height << ", " << width << std::endl;
    file << dimension_buffer.str();

    // Add array contents
    const int arr_size = num_frames * height * width;
    std::cout << arr_size << std::endl;
    for (int i = 0; i < arr_size; ++i) {
        int curr_num = arr[i];
        if (i % width == (width - 1)) {
            file << curr_num << std::endl;
        } else {
            file << curr_num << " ";
        }
    }
    file.close();
}

uint8_t* three_kernel(uint8_t *color_array, float *mask, int num_frames, int height, int width);
void tint_color(uint8_t *color_array, uint8_t color_val, float weight, int array_len);
void shade_color(uint8_t *color_array, uint8_t color_val, float weight, int array_len);

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Need to supply command line args." << std::endl;
        exit(1);
    }
    GIF_ID = std::string(argv[1]);
    MY_PATH = IS_PETER ?
        "/global/homes/p/pde23/5220Final/" : "/global/homes/a/avellm/cs5220sp24/5220Final/";

    std::ifstream r_file(MY_PATH + GIF_ID + ".red");
    if (!r_file.is_open()) std::cerr << "Cannot open red file." << std::endl;

    int num_frames, height, width;
    r_file >> num_frames;
    r_file.ignore(1);
    r_file >> height;
    r_file.ignore(1);
    r_file >> width;

    uint8_t *red_array = (uint8_t*) malloc(num_frames * height * width * sizeof(uint8_t));
    uint8_t *green_array = (uint8_t*) malloc(num_frames * height * width * sizeof(uint8_t));
    uint8_t *blue_array = (uint8_t*) malloc(num_frames * height * width * sizeof(uint8_t));


    std::string word;
    int i = 0;
    while (r_file >> word){
        red_array[i] = stoi(word);
        i++;
    }
    r_file.close();

    std::ifstream g_file(MY_PATH + GIF_ID + ".green");
    if (!g_file.is_open()) std::cerr << "Cannot open green file." << std::endl;
    g_file.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
    i = 0;
    while (g_file >> word){
        green_array[i] = stoi(word);
        i++;
    }
    g_file.close();

    std::ifstream b_file(MY_PATH + GIF_ID + ".blue");
    if (!b_file.is_open()) std::cerr << "Cannot open blue file." << std::endl;
    b_file.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
    i = 0;
    while (b_file >> word){
        blue_array[i] = stoi(word);
        i++;
    }
    b_file.close();


    // uint8_t* red_gpu, green_gpu, blue_bpu;
    // cudaMalloc((void**)&red_gpu, num_frames * height * width * sizeof(uint8_t));
    // cudaMalloc((void**)&green_gpu, num_frames * height * width * sizeof(uint8_t));
    // cudaMalloc((void**)&blue_gpu, num_frames * height * width * sizeof(uint8_t));

    // cudaMemcpy(red_gpu, red_array, num_frames * height * width * sizeof(uint8_t), cudaMemcpyHostToDevice);
    // cudaMemcpy(green_gpu, green_array, num_frames * height * width * sizeof(uint8_t), cudaMemcpyHostToDevice);
    // cudaMemcpy(blue_gpu, blue_array, num_frames * height * width * sizeof(uint8_t), cudaMemcpyHostToDevice);

    // float mask[] = {0.0, -1.0, 0.0, -1.0, 0.5, -1.0, 0.0, -1.0, 0.0};
    // red_array = three_kernel(red_array, mask, num_frames, height, width);
    // green_array = three_kernel(green_array, mask, num_frames, height, width);
    // blue_array = three_kernel(blue_array, mask, num_frames, height, width);


    output_array(red_array, "red", num_frames, height, width);
    output_array(green_array, "green", num_frames, height, width);
    output_array(blue_array, "blue", num_frames, height, width);
    

    return 0;
}