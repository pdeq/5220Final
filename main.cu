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

#define NUM_THREADS 256
#define IS_PETER false
std::string MY_PATH;
std::string GIF_ID;
int blks;


void output_array(int* arr, std::string color, int num_frames, int height, int width) {
    std::ofstream file(MY_PATH + GIF_ID + "-modified." + color);
    if (!file.is_open()) std::cerr << "Cannot open output file." << std::endl;

    // Add dimensions to top of file
    std::ostringstream dimension_buffer;
    dimension_buffer << num_frames << ", " << height << ", " << width << std::endl;
    file << dimension_buffer.str();

    // Add array contents
    const int arr_size = num_frames * height * width;
    for (int i = 0; i < arr_size; ++i) {
        int curr_num = arr[i];
        curr_num = max(curr_num, 0);
        curr_num = min(curr_num, 255);
        if (i % width == (width - 1)) {
            file << curr_num << std::endl;
        } else {
            file << curr_num << " ";
        }
    }
    file.close();
}

int* three_kernel(int *color_array, float *mask, int num_frames, int height, int width);
void tint_color(int *color_array, int color_val, float weight, int array_len);
void shade_color(int *color_array, int color_val, float weight, int array_len);
__global__ void d_tint_color(int* d_color_array, int color_val, float weight, int array_len);

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

    int *red_array = (int*) malloc(num_frames * height * width * sizeof(int));
    int *green_array = (int*) malloc(num_frames * height * width * sizeof(int));
    int *blue_array = (int*) malloc(num_frames * height * width * sizeof(int));


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


    int *d_red;
    int *d_green;
    int *d_blue;

    cudaMalloc((void**)&d_red, num_frames * height * width * sizeof(int));
    cudaMalloc((void**)&d_green, num_frames * height * width * sizeof(int));
    cudaMalloc((void**)&d_blue, num_frames * height * width * sizeof(int));

    cudaMemcpy(d_red, red_array, num_frames * height * width * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_green, green_array, num_frames * height * width * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_blue, blue_array, num_frames * height * width * sizeof(int), cudaMemcpyHostToDevice);

    blks = (num_frames * height * width + NUM_THREADS - 1) / NUM_THREADS;
    d_tint_color<<<blks, NUM_THREADS>>>(d_red, 120, 0.9, num_frames * height * width);
    d_tint_color<<<blks, NUM_THREADS>>>(d_green, 200, 0.3, num_frames * height * width);
    d_tint_color<<<blks, NUM_THREADS>>>(d_blue, 90, 0.1, num_frames * height * width);
    
    cudaMemcpy(red_array, d_red, num_frames * height * width * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(green_array, d_red, num_frames * height * width * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(blue_array, d_red, num_frames * height * width * sizeof(int), cudaMemcpyDeviceToHost);

    // float mask[] = {0.0, -1.0, 0.0, -1.0, 0.5, -1.0, 0.0, -1.0, 0.0};
    // red_array = three_kernel(red_array, mask, num_frames, height, width);
    // green_array = three_kernel(green_array, mask, num_frames, height, width);
    // blue_array = three_kernel(blue_array, mask, num_frames, height, width);

    // tint_color(red_array, 120, 0.9, num_frames * height * width);
    // tint_color(green_array, 200, 0.3, num_frames * height * width);
    // tint_color(blue_array, 90, 0.1, num_frames * height * width);


    // Copy back to arrays

    output_array(red_array, "red", num_frames, height, width);
    output_array(green_array, "green", num_frames, height, width);
    output_array(blue_array, "blue", num_frames, height, width);
    

    return 0;
}