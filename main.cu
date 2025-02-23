// main.cu
#include <algorithm>
#include <chrono>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <sstream>
#include <vector>

// Define escape codes for colors
#define RESET   "\033[0m"
#define RED     "\033[31m"
#define GREEN   "\033[32m"
#define YELLOW  "\033[33m"
#define BLUE    "\033[34m"

// ======================================================================
// Utility: Error Checking
// ======================================================================
#define CUDA_CHECK(cmd) { cudaError_t error = cmd; if(error != cudaSuccess) { std::cerr << "CUDA error: " << cudaGetErrorString(error) << " at line " << __LINE__ << "\n"; exit(1); }}

// ======================================================================
// Utility: Print CUDA Specifications
// ======================================================================
void printCudaInfo() {
    int deviceCount;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    if (err != cudaSuccess) {
        std::cout << "CUDA error: " << cudaGetErrorString(err) << "\n";
        return;
    }
    std::cout << RED << "========== CUDA Device Info ==========\n";
    for (int i = 0; i < deviceCount; i++) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        std::cout << "Device " << i << ": " << prop.name << "\n";
        std::cout << "  Compute capability: " << prop.major << "." << prop.minor << "\n";
        std::cout << "  Total global memory: " << (prop.totalGlobalMem / (1024 * 1024)) << " MB\n";
        std::cout << "  Multiprocessors: " << prop.multiProcessorCount << "\n";
        std::cout << "  Max threads per block: " << prop.maxThreadsPerBlock << "\n";
    }
    std::cout << "======================================\n\n";
}

// ======================================================================
// Utility: Compute Required Trials as String
// ======================================================================
std::string computeRequiredTrialsStrMC(int precision) {
    std::ostringstream oss;
    oss << "3e" << (2 * precision);
    return oss.str();
}

// ======================================================================
// Utility: Compute Actual Trials
// ======================================================================
unsigned long long computeRequiredTrialsMC(int precision) {
    return static_cast<unsigned long long>(3 * std::pow(10, precision - 2)); // static_cast<unsigned long long>(3 * std::pow(10, 2 * precision));
}

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

// Multi-threaded using CUDA with stratified sampling by x–coordinate
long double monte_carlo_pi_multithreaded_using_cuda_with_stratified_x(unsigned long long trials, int num_strata) {
    unsigned long long h_inside = 0;
    unsigned long long *d_inside;
    cudaMalloc(&d_inside, sizeof(unsigned long long));
    cudaMemset(d_inside, 0, sizeof(unsigned long long));

    int threadsPerBlock = 256;
    int blocks = 1024;

    for (int s = 0; s < num_strata; ++s) {
        double x_min = s * 1.0 / num_strata;
        double x_max = (s + 1) * 1.0 / num_strata;
        double y_min = 0;
        double y_max = 1;

        monte_carlo_kernel<<<blocks, threadsPerBlock>>>(trials / num_strata, x_min, x_max, y_min, y_max, d_inside);
    }

    cudaDeviceSynchronize();
    cudaMemcpy(&h_inside, d_inside, sizeof(unsigned long long), cudaMemcpyDeviceToHost);
    cudaFree(d_inside);

    return 4.0 * h_inside / trials;
}

// Multi-threaded using CUDA with grid–based stratified sampling
long double monte_carlo_pi_multithreaded_using_cuda_with_stratified_grid(unsigned long long trials, int grid_dim) {
    unsigned long long h_inside = 0;
    unsigned long long *d_inside;
    cudaMalloc(&d_inside, sizeof(unsigned long long));
    cudaMemset(d_inside, 0, sizeof(unsigned long long));

    int threadsPerBlock = 256;
    int blocks = 1024;

    for (int i = 0; i < grid_dim; ++i) {
        for (int j = 0; j < grid_dim; ++j) {
            double x_min = i * 1.0 / grid_dim;
            double x_max = (i + 1) * 1.0 / grid_dim;
            double y_min = j * 1.0 / grid_dim;
            double y_max = (j + 1) * 1.0 / grid_dim;
            unsigned long long seed_offset = i * grid_dim + j;

            monte_carlo_kernel<<<blocks, threadsPerBlock>>>(trials / (grid_dim * grid_dim), x_min, x_max, y_min, y_max, d_inside, seed_offset);
        }
    }

    cudaDeviceSynchronize();
    cudaMemcpy(&h_inside, d_inside, sizeof(unsigned long long), cudaMemcpyDeviceToHost);
    cudaFree(d_inside);

    return 4.0 * h_inside / trials;
}

// ======================================================================
// Main Function
// ======================================================================
int main() {
    printCudaInfo();

    // Append results to file
    std::ofstream outFilePrecisionsMC("results_precisions_mc.csv", std::ios_base::app);
    if (!outFilePrecisionsMC) {
        std::cerr << "Error opening file for appending.\n";
        return 1;
    }

    // List of target precisions: 5, 10, 15, and 20 decimal places.
    std::vector<int> precisions = {5, 10, 15, 20};

    // For demonstration we limit the simulation trials.
    const unsigned long long max_simulation_trials = 10000000ULL; // 10 million trials

    unsigned int num_threads = 16;

    for (int precision : precisions) {
        std::cout << GREEN << "Precision: " << precision << "\n\n";

        std::string req_trials_str = computeRequiredTrialsStrMC(precision);
        std::cout << BLUE << "Required trials for " << precision << " decimal places: " 
                  << req_trials_str << "\n";
        std::cout << "Decimal precision set to: " << precision << " places\n";

        // // For simulation, we use the maximum allowed trials (for demo purposes)
        // unsigned long long trials = max_simulation_trials;
        // std::cout << "Using " << trials << " trials for simulation.\n\n";
        // Use actual trials computed from precision instead of a fixed maximum.
        unsigned long long trials = computeRequiredTrialsMC(precision);
        std::cout << "Using " << trials << " trials for simulation.\n\n";

        const int iterationsCount = 5;

        // Set output precision for simulation results
        std::cout << RESET << std::fixed << std::setprecision(precision);

        // ---------------------------------------------------
        // Monte Carlo: Multi-threaded using CUDA with uniform sampling
        Stats mcMultiThreadedCUDAUniformStats = measureSimulation(iterationsCount, monte_carlo_pi_multithreaded_using_cuda_with_uniform, trials);
        std::cout << "Monte Carlo Multi-threaded using CUDA with Uniform Sampling:\n"
              << "  Average PI Value: " << mcMultiThreadedCUDAUniformStats.avgValue << "\n"
              << "  Average Time:     " << mcMultiThreadedCUDAUniformStats.avgTime << " s\n"
              << "  25th Percentile:  " << mcMultiThreadedCUDAUniformStats.p25 << " s\n"
              << "  Median Time:      " << mcMultiThreadedCUDAUniformStats.median << " s\n"
              << "  75th Percentile:  " << mcMultiThreadedCUDAUniformStats.p75 << " s\n\n";
        outFilePrecisionsMC << precision << ",Monte Carlo Multi-threaded using CUDA with Uniform Sampling,"
                << std::fixed << std::setprecision(precision)
                << mcMultiThreadedCUDAUniformStats.avgValue << ","
                << mcMultiThreadedCUDAUniformStats.avgTime << ","
                << mcMultiThreadedCUDAUniformStats.p25 << ","
                << mcMultiThreadedCUDAUniformStats.median << ","
                << mcMultiThreadedCUDAUniformStats.p75 << "\n";

        // ---------------------------------------------------
        // Monte Carlo: Multi-threaded using CUDA with stratified sampling by x–coordinate
        Stats mcMultiThreadedCUDAStratifiedXStats = measureSimulation(iterationsCount, monte_carlo_pi_multithreaded_using_cuda_with_stratified_x, trials, num_threads);
        std::cout << "Monte Carlo Multi-threaded using CUDA with Stratified Sampling by x-Coordinate:\n"
              << "  Average PI Value: " << mcMultiThreadedCUDAStratifiedXStats.avgValue << "\n"
              << "  Average Time:     " << mcMultiThreadedCUDAStratifiedXStats.avgTime << " s\n"
              << "  25th Percentile:  " << mcMultiThreadedCUDAStratifiedXStats.p25 << " s\n"
              << "  Median Time:      " << mcMultiThreadedCUDAStratifiedXStats.median << " s\n"
              << "  75th Percentile:  " << mcMultiThreadedCUDAStratifiedXStats.p75 << " s\n\n";
        outFilePrecisionsMC << precision << ",Monte Carlo Multi-threaded using CUDA with Stratified Sampling by x-Coordinate,"
                << std::fixed << std::setprecision(precision)
                << mcMultiThreadedCUDAStratifiedXStats.avgValue << ","
                << mcMultiThreadedCUDAStratifiedXStats.avgTime << ","
                << mcMultiThreadedCUDAStratifiedXStats.p25 << ","
                << mcMultiThreadedCUDAStratifiedXStats.median << ","
                << mcMultiThreadedCUDAStratifiedXStats.p75 << "\n";

        // ---------------------------------------------------
        // Monte Carlo: Multi-threaded using CUDA with grid–based stratified sampling
        Stats mcMultiThreadedCUDAStratifiedGridStats = measureSimulation(iterationsCount, monte_carlo_pi_multithreaded_using_cuda_with_stratified_grid, trials, sqrt(num_threads));
        std::cout << "Monte Carlo Multi-threaded using CUDA with Grid-based Stratified Sampling:\n"
              << "  Average PI Value: " << mcMultiThreadedCUDAStratifiedGridStats.avgValue << "\n"
              << "  Average Time:     " << mcMultiThreadedCUDAStratifiedGridStats.avgTime << " s\n"
              << "  25th Percentile:  " << mcMultiThreadedCUDAStratifiedGridStats.p25 << " s\n"
              << "  Median Time:      " << mcMultiThreadedCUDAStratifiedGridStats.median << " s\n"
              << "  75th Percentile:  " << mcMultiThreadedCUDAStratifiedGridStats.p75 << " s\n\n";
        outFilePrecisionsMC << precision << ",Monte Carlo Multi-threaded using CUDA with Grid-based Stratified Sampling,"
                << std::fixed << std::setprecision(precision)
                << mcMultiThreadedCUDAStratifiedGridStats.avgValue << ","
                << mcMultiThreadedCUDAStratifiedGridStats.avgTime << ","
                << mcMultiThreadedCUDAStratifiedGridStats.p25 << ","
                << mcMultiThreadedCUDAStratifiedGridStats.median << ","
                << mcMultiThreadedCUDAStratifiedGridStats.p75 << "\n";

        std::cout << "------------------------------------------------------\n\n";
    }

    outFilePrecisionsMC.close();

    // Append results to file
    std::ofstream outFileTrials("results_trials.csv", std::ios_base::app);
    if (!outFileTrials) {
        std::cerr << "Error opening file for appending.\n";
        return 1;
    }

    // Define the fixed trial counts.
    std::vector<unsigned long long> trialCounts = { (1ULL << 24), (1ULL << 26), (1ULL << 28) };

    int precision = 16;

    for (unsigned long long trials : trialCounts) {
        std::cout << GREEN << "Trials: " << trials << "\n\n";

        const int iterationsCount = 5;

        // Set output precision for simulation results
        std::cout << RESET << std::fixed << std::setprecision(precision);

        // ---------------------------------------------------
        // Monte Carlo: Multi-threaded using CUDA with uniform sampling
        Stats mcMultiThreadedCUDAUniformStats = measureSimulation(iterationsCount, monte_carlo_pi_multithreaded_using_cuda_with_uniform, trials);
        std::cout << "Monte Carlo Multi-threaded using CUDA with Uniform Sampling:\n"
              << "  Average PI Value: " << mcMultiThreadedCUDAUniformStats.avgValue << "\n"
              << "  Average Time:     " << mcMultiThreadedCUDAUniformStats.avgTime << " s\n"
              << "  25th Percentile:  " << mcMultiThreadedCUDAUniformStats.p25 << " s\n"
              << "  Median Time:      " << mcMultiThreadedCUDAUniformStats.median << " s\n"
              << "  75th Percentile:  " << mcMultiThreadedCUDAUniformStats.p75 << " s\n\n";
        outFilePrecisionsMC << precision << ",Monte Carlo Multi-threaded using CUDA with Uniform Sampling,"
                << std::fixed << std::setprecision(precision)
                << mcMultiThreadedCUDAUniformStats.avgValue << ","
                << mcMultiThreadedCUDAUniformStats.avgTime << ","
                << mcMultiThreadedCUDAUniformStats.p25 << ","
                << mcMultiThreadedCUDAUniformStats.median << ","
                << mcMultiThreadedCUDAUniformStats.p75 << "\n";

        // ---------------------------------------------------
        // Monte Carlo: Multi-threaded using CUDA with stratified sampling by x–coordinate
        Stats mcMultiThreadedCUDAStratifiedXStats = measureSimulation(iterationsCount, monte_carlo_pi_multithreaded_using_cuda_with_stratified_x, trials, num_threads);
        std::cout << "Monte Carlo Multi-threaded using CUDA with Stratified Sampling by x-Coordinate:\n"
              << "  Average PI Value: " << mcMultiThreadedCUDAStratifiedXStats.avgValue << "\n"
              << "  Average Time:     " << mcMultiThreadedCUDAStratifiedXStats.avgTime << " s\n"
              << "  25th Percentile:  " << mcMultiThreadedCUDAStratifiedXStats.p25 << " s\n"
              << "  Median Time:      " << mcMultiThreadedCUDAStratifiedXStats.median << " s\n"
              << "  75th Percentile:  " << mcMultiThreadedCUDAStratifiedXStats.p75 << " s\n\n";
        outFilePrecisionsMC << precision << ",Monte Carlo Multi-threaded using CUDA with Stratified Sampling by x-Coordinate,"
                << std::fixed << std::setprecision(precision)
                << mcMultiThreadedCUDAStratifiedXStats.avgValue << ","
                << mcMultiThreadedCUDAStratifiedXStats.avgTime << ","
                << mcMultiThreadedCUDAStratifiedXStats.p25 << ","
                << mcMultiThreadedCUDAStratifiedXStats.median << ","
                << mcMultiThreadedCUDAStratifiedXStats.p75 << "\n";

        // ---------------------------------------------------
        // Monte Carlo: Multi-threaded using CUDA with grid–based stratified sampling
        Stats mcMultiThreadedCUDAStratifiedGridStats = measureSimulation(iterationsCount, monte_carlo_pi_multithreaded_using_cuda_with_stratified_grid, trials, sqrt(num_threads));
        std::cout << "Monte Carlo Multi-threaded using CUDA with Grid-based Stratified Sampling:\n"
              << "  Average PI Value: " << mcMultiThreadedCUDAStratifiedGridStats.avgValue << "\n"
              << "  Average Time:     " << mcMultiThreadedCUDAStratifiedGridStats.avgTime << " s\n"
              << "  25th Percentile:  " << mcMultiThreadedCUDAStratifiedGridStats.p25 << " s\n"
              << "  Median Time:      " << mcMultiThreadedCUDAStratifiedGridStats.median << " s\n"
              << "  75th Percentile:  " << mcMultiThreadedCUDAStratifiedGridStats.p75 << " s\n\n";
        outFilePrecisionsMC << precision << ",Monte Carlo Multi-threaded using CUDA with Grid-based Stratified Sampling,"
                << std::fixed << std::setprecision(precision)
                << mcMultiThreadedCUDAStratifiedGridStats.avgValue << ","
                << mcMultiThreadedCUDAStratifiedGridStats.avgTime << ","
                << mcMultiThreadedCUDAStratifiedGridStats.p25 << ","
                << mcMultiThreadedCUDAStratifiedGridStats.median << ","
                << mcMultiThreadedCUDAStratifiedGridStats.p75 << "\n";

        std::cout << "------------------------------------------------------\n\n";
    }

    outFileTrials.close();

    return 0;
}
