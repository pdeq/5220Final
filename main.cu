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

    std::string option(argv[2]);
    blks = (num_frames * height * width + NUM_THREADS - 1) / NUM_THREADS;

    int *d_red, *d_green, *d_blue;
    cudaMalloc((void**)&d_red, num_frames * height * width * sizeof(int));
    cudaMalloc((void**)&d_green, num_frames * height * width * sizeof(int));
    cudaMalloc((void**)&d_blue, num_frames * height * width * sizeof(int));

    cudaMemcpy(d_red, red_array, num_frames * height * width * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_green, green_array, num_frames * height * width * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_blue, blue_array, num_frames * height * width * sizeof(int), cudaMemcpyHostToDevice);

    auto start_time = std::chrono::steady_clock::now();
    if (option == "tint") {
        // for (int j = 0; j < 100000; ++j) {
            d_tint_color<<<blks, NUM_THREADS>>>(d_red, 120, 0.75, num_frames * height * width);
            d_tint_color<<<blks, NUM_THREADS>>>(d_green, 200, 0.25, num_frames * height * width);
            d_tint_color<<<blks, NUM_THREADS>>>(d_blue, 100, 0.5, num_frames * height * width);
        // }
        
        cudaMemcpy(red_array, d_red, num_frames * height * width * sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(green_array, d_green, num_frames * height * width * sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(blue_array, d_blue, num_frames * height * width * sizeof(int), cudaMemcpyDeviceToHost);
        
    } else if (option == "mask") {
        float mask[] = {-1, -1, -1, -1, 8, -1, -1, -1, -1};
        float *d_mask;
        cudaMalloc((void**)&d_mask, 9 * sizeof(float));
        cudaMemcpy(d_mask, mask, 9 * sizeof(float), cudaMemcpyHostToDevice);

        int *d_masked_red, *d_masked_green, *d_masked_blue;
        cudaMalloc((void**)&d_masked_red, num_frames * height * width * sizeof(int));
        cudaMalloc((void**)&d_masked_green, num_frames * height * width * sizeof(int));
        cudaMalloc((void**)&d_masked_blue, num_frames * height * width * sizeof(int));

        // for (int j = 0; j < 100000; ++j) {
            d_mask3<<<blks, NUM_THREADS>>>(d_red, d_masked_red, d_mask, num_frames, height, width);
            d_mask3<<<blks, NUM_THREADS>>>(d_green, d_masked_green, d_mask, num_frames, height, width);
            d_mask3<<<blks, NUM_THREADS>>>(d_blue, d_masked_blue, d_mask, num_frames, height, width);
        // }
        
        cudaMemcpy(red_array, d_masked_red, num_frames * height * width * sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(green_array, d_masked_green, num_frames * height * width * sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(blue_array, d_masked_blue, num_frames * height * width * sizeof(int), cudaMemcpyDeviceToHost);

    }

    cudaDeviceSynchronize();
    auto end_time = std::chrono::steady_clock::now();
    std::chrono::duration<double> diff = end_time - start_time;
    double seconds = diff.count();
    std::cout << option << " took " << seconds <<
        " on " << num_frames * height * width <<
        " particles." << std::endl;

    output_array(red_array, "red", num_frames, height, width);
    output_array(green_array, "green", num_frames, height, width);
    output_array(blue_array, "blue", num_frames, height, width);
    
    return 0;
}