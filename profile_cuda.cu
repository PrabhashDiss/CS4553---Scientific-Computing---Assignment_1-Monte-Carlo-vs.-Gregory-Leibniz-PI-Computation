// profile_cuda.cu
#include <algorithm>
#include <chrono>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <vector>

// ======================================================================
// Utility: Error Checking
// ======================================================================
#define CUDA_CHECK(cmd) { cudaError_t error = cmd; if(error != cudaSuccess) { std::cerr << "CUDA error: " << cudaGetErrorString(error) << " at line " << __LINE__ << "\n"; exit(1); }}

// ======================================================================
// Utility: Generic Measurement Function
// ======================================================================
// Structure to hold the statistics.
struct Stats {
    double avgTime;
    double p25;
    double median;
    double p75;
    long double avgValue;
};

// Generic measurement function that accepts a simulation function and its arguments.
template<typename Func, typename... Args>
Stats measureSimulation(int iterations, Func simulation, Args&&... args) {
    Stats stats;
    std::vector<double> times;
    std::vector<long double> values;

    for (int i = 0; i < iterations; i++) {
        auto start = std::chrono::high_resolution_clock::now();
        long double result = simulation(std::forward<Args>(args)...);
        auto end = std::chrono::high_resolution_clock::now();
        double elapsed = std::chrono::duration<double>(end - start).count();
        times.push_back(elapsed);
        values.push_back(result);
    }

    // Compute average value.
    stats.avgValue = std::accumulate(values.begin(), values.end(), 0.0) / values.size();
    // Compute average time.
    stats.avgTime = std::accumulate(times.begin(), times.end(), 0.0) / times.size();

    // Compute percentiles for times.
    std::vector<double> sortedTimes = times;
    std::sort(sortedTimes.begin(), sortedTimes.end());
    size_t n = sortedTimes.size();
    stats.p25 = sortedTimes[static_cast<size_t>(0.25 * n)];
    stats.median = sortedTimes[static_cast<size_t>(0.5 * n)];
    stats.p75 = sortedTimes[static_cast<size_t>(0.75 * n)];

    return stats;
}

// ======================================================================
// Kernel
// ======================================================================
__global__ void monte_carlo_kernel(unsigned long long trials,
                    double x_min, double x_max,
                    double y_min, double y_max,
                    unsigned long long *d_inside,
                    unsigned long long seed_offset = 0) {
    unsigned long long tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long stride = blockDim.x * gridDim.x;
    unsigned long long count = 0;
    unsigned long long seed = tid + seed_offset;
    curandState state;
    curand_init(seed, tid, 0, &state);
    for (unsigned long long i = tid; i < trials; i += stride) {
        double x = x_min + (x_max - x_min) * curand_uniform_double(&state);
        double y = y_min + (y_max - y_min) * curand_uniform_double(&state);
        if (x*x + y*y <= 1.0)
            count++;
    }
    atomicAdd(d_inside, count);
}

// Multi-threaded using CUDA with uniform sampling
long double monte_carlo_pi_multithreaded_using_cuda_with_uniform(unsigned long long trials) {
    unsigned long long h_inside = 0;
    unsigned long long *d_inside;
    cudaMalloc(&d_inside, sizeof(unsigned long long));
    cudaMemset(d_inside, 0, sizeof(unsigned long long));

    int threadsPerBlock = 256;
    int blocks = 1024;
    monte_carlo_kernel<<<blocks, threadsPerBlock>>>(trials, 0, 1, 0, 1, d_inside);
    cudaDeviceSynchronize();
    cudaMemcpy(&h_inside, d_inside, sizeof(unsigned long long), cudaMemcpyDeviceToHost);
    cudaFree(d_inside);

    return 4.0 * h_inside / trials;
}

// ======================================================================
// Main Function
// ======================================================================
int main() {
    int precision = 16;

    const unsigned long long max_simulation_trials = 10000000ULL;
    unsigned long long trials = max_simulation_trials;

    const int iterationsCount = 5;

    unsigned int num_threads = 16;

    // Set output precision for simulation results
    std::cout << std::fixed << std::setprecision(precision);

    // ---------------------------------------------------
    // Monte Carlo: Multi-threaded using CUDA with uniform sampling
    Stats mcMultiThreadedCUDAUniformStats = measureSimulation(iterationsCount, monte_carlo_pi_multithreaded_using_cuda_with_uniform, trials);
    std::cout << "Monte Carlo Multi-threaded using CUDA with Uniform Sampling:\n"
            << "  Average PI Value: " << mcMultiThreadedCUDAUniformStats.avgValue << "\n"
            << "  Average Time:     " << mcMultiThreadedCUDAUniformStats.avgTime << " s\n"
            << "  25th Percentile:  " << mcMultiThreadedCUDAUniformStats.p25 << " s\n"
            << "  Median Time:      " << mcMultiThreadedCUDAUniformStats.median << " s\n"
            << "  75th Percentile:  " << mcMultiThreadedCUDAUniformStats.p75 << " s\n\n";

    return 0;
}
