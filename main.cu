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
#define IS_PETER true
std::string MY_PATH;
std::string GIF_ID;
int blks;


void output_array(int* arr, std::string color, int num_frames, int height, int width, int duration) {
    std::ofstream file(MY_PATH + GIF_ID + "-modified." + color);
    if (!file.is_open()) std::cerr << "Cannot open output file." << std::endl;

    // Add dimensions to top of file
    std::ostringstream dimension_buffer;
    dimension_buffer << num_frames << ", " << height << ", " << width << ", " << duration << std::endl;
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

    int num_frames, height, width, duration;
    r_file >> num_frames;
    r_file.ignore(1);
    r_file >> height;
    r_file.ignore(1);
    r_file >> width;
    r_file.ignore(1);
    r_file >> duration;

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
    
    int NUM_FRAMES = num_frames;
    int DURATION = duration;

    auto start_time = std::chrono::steady_clock::now();
    if (option == "tint") {
        // for (int j = 0; j < 100000; ++j) {
            d_tint_color<<<blks, NUM_THREADS>>>(d_red, 120, 1.0, num_frames * height * width);
            d_tint_color<<<blks, NUM_THREADS>>>(d_green, 200, 1.0, num_frames * height * width);
            d_tint_color<<<blks, NUM_THREADS>>>(d_blue, 100, 1.0, num_frames * height * width);
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

    } else if (option == "median"){
        int *d_masked_red, *d_masked_green, *d_masked_blue;
        cudaMalloc((void**)&d_masked_red, num_frames * height * width * sizeof(int));
        cudaMalloc((void**)&d_masked_green, num_frames * height * width * sizeof(int));
        cudaMalloc((void**)&d_masked_blue, num_frames * height * width * sizeof(int));

        // for (int j = 0; j < 100000; ++j) {
            d_median3<<<blks, NUM_THREADS>>>(d_red, d_masked_red, num_frames, height, width);
            d_median3<<<blks, NUM_THREADS>>>(d_green, d_masked_green, num_frames, height, width);
            d_median3<<<blks, NUM_THREADS>>>(d_blue, d_masked_blue, num_frames, height, width);
        // }

        cudaMemcpy(red_array, d_masked_red, num_frames * height * width * sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(green_array, d_masked_green, num_frames * height * width * sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(blue_array, d_masked_blue, num_frames * height * width * sizeof(int), cudaMemcpyDeviceToHost);
    } else if (option == "interpolate") {
        int *d_interp_red, *d_interp_green, *d_interp_blue;
        int interp_pixel_amt = 2 * num_frames * height * width - 1;
        cudaMalloc((void**)&d_interp_red, interp_pixel_amt * sizeof(int));
        cudaMalloc((void**)&d_interp_green, interp_pixel_amt * sizeof(int));
        cudaMalloc((void**)&d_interp_blue, interp_pixel_amt * sizeof(int));

        d_interpolate<<<blks, NUM_THREADS>>>(d_red, d_interp_red, num_frames, height, width);
        d_interpolate<<<blks, NUM_THREADS>>>(d_green, d_interp_green, num_frames, height, width);
        d_interpolate<<<blks, NUM_THREADS>>>(d_blue, d_interp_blue, num_frames, height, width);

        red_array = (int *) malloc(interp_pixel_amt * sizeof(int));
        green_array = (int *) malloc(interp_pixel_amt * sizeof(int));
        blue_array = (int *) malloc(interp_pixel_amt * sizeof(int));

        cudaMemcpy(red_array, d_interp_red, interp_pixel_amt * sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(green_array, d_interp_green, interp_pixel_amt * sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(blue_array, d_interp_blue, interp_pixel_amt * sizeof(int), cudaMemcpyDeviceToHost);
        NUM_FRAMES = 2 * num_frames - 1;
        DURATION = (duration * num_frames) / (2 * num_frames - 1);
    } else if (option == "gray") {
        int *d_gray;
        cudaMalloc((void**)&d_gray, num_frames * height * width * sizeof(int));

        d_gray_scale<<<blks, NUM_THREADS>>>(d_gray, d_red, d_green, d_blue, num_frames, height, width);

        cudaMemcpy(red_array, d_gray, num_frames * height * width * sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(green_array, d_gray, num_frames * height * width * sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(blue_array, d_gray, num_frames * height * width * sizeof(int), cudaMemcpyDeviceToHost); 
    } else if (option == "paint") {
        std::mt19937 generator(std::random_device{}());
        std::normal_distribution<> distribution(0, 20);
        int *gaussian = (int *) malloc(height * width * sizeof(int));
        for (int g = 0; g < height * width; ++g){
            gaussian[g] = (int) distribution(generator);
        }

        int *d_gaussian;
        cudaMalloc((void**)&d_gaussian, height * width * sizeof(int));
        cudaMemcpy(d_gaussian, gaussian, height * width * sizeof(int), cudaMemcpyHostToDevice);

        d_gauss<<<blks, NUM_THREADS>>>(d_red, num_frames, height, width, d_gaussian);
        d_gauss<<<blks, NUM_THREADS>>>(d_green, num_frames, height, width, d_gaussian);
        d_gauss<<<blks, NUM_THREADS>>>(d_blue, num_frames, height, width, d_gaussian);

        free(gaussian);

        // cudaMemcpy(red_array, d_red, num_frames * height * width * sizeof(int), cudaMemcpyDeviceToHost);
        // cudaMemcpy(green_array, d_green, num_frames * height * width * sizeof(int), cudaMemcpyDeviceToHost);
        // cudaMemcpy(blue_array, d_blue, num_frames * height * width * sizeof(int), cudaMemcpyDeviceToHost);
        float mask[] = {-1, -1, -1, -1, 8, -1, -1, -1, -1};
        float *d_mask;
        cudaMalloc((void**)&d_mask, 9 * sizeof(float));
        cudaMemcpy(d_mask, mask, 9 * sizeof(float), cudaMemcpyHostToDevice);

        int *d_masked_red, *d_masked_green, *d_masked_blue;
        cudaMalloc((void**)&d_masked_red, num_frames * height * width * sizeof(int));
        cudaMalloc((void**)&d_masked_green, num_frames * height * width * sizeof(int));
        cudaMalloc((void**)&d_masked_blue, num_frames * height * width * sizeof(int));

        d_mask3<<<blks, NUM_THREADS>>>(d_red, d_masked_red, d_mask, num_frames, height, width);
        d_mask3<<<blks, NUM_THREADS>>>(d_green, d_masked_green, d_mask, num_frames, height, width);
        d_mask3<<<blks, NUM_THREADS>>>(d_blue, d_masked_blue, d_mask, num_frames, height, width);

        cudaMemcpy(d_red, red_array, height * width * num_frames * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_green, green_array, height * width * num_frames * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_blue, blue_array, height * width * num_frames * sizeof(int), cudaMemcpyHostToDevice);

        d_imgmul<<<blks, NUM_THREADS>>>(d_red, d_masked_red, num_frames * height * width);
        d_imgmul<<<blks, NUM_THREADS>>>(d_green, d_masked_green, num_frames * height * width);
        d_imgmul<<<blks, NUM_THREADS>>>(d_blue, d_masked_blue, num_frames * height * width);

        cudaMemcpy(red_array, d_red, num_frames * height * width * sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(green_array, d_green, num_frames * height * width * sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(blue_array, d_blue, num_frames * height * width * sizeof(int), cudaMemcpyDeviceToHost);
    } else if (option == "segment"){
        std::cout << "This ``segment`` option requires additional information." << std::endl;
        int segments_amt;
        std::cout << "How many segments should be done for your GIF? ";
        std::cin >> segments_amt;
        std::cout << "Each frame of your GIF, " << GIF_ID <<
            ", is " << width << "x" << height << " pixels." << std::endl;
        std::cout << "In the following inputs, please answer as ``2,3``, for example. " <<
            "If you don't want to make a guess, enter ``-1,-1``" << std::endl;
        int x_buf, y_buf;
        int *x_centroid_locs = (int *) malloc(segments_amt * sizeof(int));
        int *y_centroid_locs = (int *) malloc(segments_amt * sizeof(int));

        std::mt19937 generator(std::random_device{}());
        std::uniform_int_distribution<int> w_dist(0, width);
        std::uniform_int_distribution<int> h_dist(0, height);

        for (int i = 0; i < segments_amt; ++i) {
            std::cout << "Give a guess where you'd like centroid " << i <<
                " to be placed: ";
            std::cin >> x_buf;
            std::cin.ignore(1);
            std::cin >> y_buf;
            x_centroid_locs[i] = x_buf > -1 ? x_buf : w_dist(generator);
            y_centroid_locs[i] = y_buf > -1 ? y_buf : h_dist(generator);
        }

        int k_count = segments_amt;
        int rs[] = {45, 120};//, 200};
        int gs[] = {90, 133};//, 57};
        int bs[] = {130, 172};//, 58};

        int *d_rs, *d_gs, *d_bs;
        cudaMalloc((void**)&d_rs, k_count * sizeof(int));
        cudaMemcpy(d_rs, rs, k_count * sizeof(int), cudaMemcpyHostToDevice);
        cudaMalloc((void**)&d_gs, k_count * sizeof(int));
        cudaMemcpy(d_gs, gs, k_count * sizeof(int), cudaMemcpyHostToDevice);
        cudaMalloc((void**)&d_bs, k_count * sizeof(int));
        cudaMemcpy(d_bs, bs, k_count * sizeof(int), cudaMemcpyHostToDevice);

        float *means = (float*) malloc(num_frames * k_count * 5 * sizeof(float));
        
        for (int v = 0; v < num_frames; ++v){
            for (int z = 0; z < k_count; ++z){
                int r_w = x_centroid_locs[z]; //w_dist(generator);
                int r_h = y_centroid_locs[z]; //h_dist(generator);
                int ind = v * (height * width) + r_h * width + r_w;
                means[v * (k_count * 5) + z * 5] = (float) red_array[ind];
                means[v * (k_count * 5) + z * 5 + 1] = (float) green_array[ind];
                means[v * (k_count * 5) + z * 5 + 2] = (float) blue_array[ind];
                means[v * (k_count * 5) + z * 5 + 3] = (float) r_w;
                means[v * (k_count * 5) + z * 5 + 4] = (float) r_h;
            }
        }

        float *d_means;
        int *d_assignments;
        int *d_count;
        
        cudaMalloc((void**)&d_assignments, num_frames * height * width * sizeof(int));
        cudaMalloc((void**)&d_means, num_frames * k_count * 5 * sizeof(float));
        cudaMalloc((void**)&d_count, num_frames * k_count * sizeof(int));

        cudaMemcpy(d_means, means, num_frames * k_count * 5 * sizeof(float), cudaMemcpyHostToDevice);

        int MAX_ITERS = 1000; // Probably bigger, but this is a start
        for (int t = 0; t < MAX_ITERS; ++t){
            cudaMemset(d_count, 0, num_frames * k_count * sizeof(int));
            cudaDeviceSynchronize();

            parallel_group<<<blks, NUM_THREADS>>>(d_red, d_green, d_blue, d_means, d_assignments, d_count, width, height, num_frames, k_count);
            cudaDeviceSynchronize();

            cudaMemset(d_means, 0.0, num_frames * k_count * 5 * sizeof(float));
            cudaDeviceSynchronize();

            parallel_means<<<blks, NUM_THREADS>>>(d_red, d_green, d_blue, d_means, d_assignments, d_count, width, height, num_frames, k_count);
            cudaDeviceSynchronize();
        }
        parallel_group<<<blks, NUM_THREADS>>>(d_red, d_green, d_blue, d_means, d_assignments, d_count, width, height, num_frames, k_count);
        cudaDeviceSynchronize();

        k_colors<<<blks, NUM_THREADS>>>(d_red, d_green, d_blue, d_assignments, d_rs, d_gs, d_bs, num_frames, height, width, k_count);

        cudaMemcpy(red_array, d_red, num_frames * height * width * sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(green_array, d_green, num_frames * height * width * sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(blue_array, d_blue, num_frames * height * width * sizeof(int), cudaMemcpyDeviceToHost);
    }

    cudaDeviceSynchronize();
    auto end_time = std::chrono::steady_clock::now();
    std::chrono::duration<double> diff = end_time - start_time;
    double seconds = diff.count();
    std::cout << option << " took " << seconds <<
        " on " << num_frames * height * width <<
        " pixels." << std::endl;
        

    output_array(red_array, "red", NUM_FRAMES, height, width, DURATION);
    output_array(green_array, "green", NUM_FRAMES, height, width, DURATION);
    output_array(blue_array, "blue", NUM_FRAMES, height, width, DURATION);
    
    return 0;
}