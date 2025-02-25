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
// CUDA Infrastructure
// ======================================================================
#define CUDA_CHECK(cmd) cudaAssert(cmd, __FILE__, __LINE__)
void cudaAssert(cudaError_t code, const char* file, int line) {
    if(code != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(code)
                  << " in " << file << ":" << line << "\n";
        exit(1);
    }
}

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
    if (precision == 5) {
        return static_cast<unsigned long long>(3 * std::pow(10, 2));
    } else if (precision == 10) {
        return static_cast<unsigned long long>(3 * std::pow(10, 4));
    } else if (precision == 15) {
        return static_cast<unsigned long long>(3 * std::pow(10, 6));
    } else if (precision == 20) {
        return static_cast<unsigned long long>(3 * std::pow(10, 8));
    } else {
        return static_cast<unsigned long long>(3 * std::pow(10, precision - 2)); // static_cast<unsigned long long>(3 * std::pow(10, 2 * precision));
    }
}

// ======================================================================
// Measurement & Reporting
// ======================================================================
// Structure to hold the statistics.
struct Stats {
    double avgTime;
    double p25;
    double median;
    double p75;
    long double avgValue;
    long double absoluteError;
};

// Function to format absolute error
std::string formatAbsoluteError(const long double& value, int precision = 10) {
    // Convert to double for formatting (this may lose some precision).
    double dVal = static_cast<double>(value);
    if (dVal == 0.0) {
        return "0";
    }
    // Compute exponent: floor(log10(|value|))
    int exponent = static_cast<int>(std::floor(std::log10(std::fabs(dVal))));
    double mantissa = dVal / std::pow(10, exponent);

    std::ostringstream oss;
    oss << std::fixed << std::setprecision(precision) << mantissa;
    oss << " * 10^" << exponent;
    return oss.str();
}

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

    // Define a high-precision true value of π.
    long double true_pi(3.141592653589793238462643383279502884197);

    // Calculate the absolute error.
    stats.absoluteError = (stats.avgValue >= true_pi) ? (stats.avgValue - true_pi)
                                                     : (true_pi - stats.avgValue);

    return stats;
}

void report_results(const std::string& filename, auto identifier, auto precision, const Stats& stats) {
    std::cout << identifier << ":\n"
              << std::fixed << std::setprecision(precision)
              << "  Average PI Value: " << stats.avgValue << "\n"
              << "  Absolute Error: " << formatAbsoluteError(stats.absoluteError, precision) << "\n"
              << "  Average Time: " << stats.avgTime << "\n"
              << "  25th Percentile: " << stats.p25 << "\n"
              << "  Median Time: " << stats.median << "\n"
              << "  75th Percentile: " << stats.p75 << "\n\n";
    std::ofstream file(filename, std::ios_base::app);
    file << std::fixed << std::setprecision(precision)
         << identifier << "," << stats.avgValue << ","
         << formatAbsoluteError(stats.absoluteError, precision) << ","
         << stats.avgTime << "," << stats.p25 << ","
         << stats.median << "," << stats.p75 << "\n";
}

// ======================================================================
// Core CUDA Kernels
// ======================================================================
__global__ void monte_carlo_kernel(unsigned long long trials,
                    double x_min, double x_max,
                    double y_min, double y_max,
                    unsigned long long *d_inside,
                    unsigned long long seed_offset = 0) {
    unsigned long long tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long stride = blockDim.x * gridDim.x;
    unsigned long long count = 0;

    curandState state;
    curand_init(tid + seed_offset, 0, 0, &state);
    
    for (unsigned long long i = tid; i < trials; i += stride) {
        double x = x_min + (x_max - x_min) * curand_uniform_double(&state);
        double y = y_min + (y_max - y_min) * curand_uniform_double(&state);
        count += (x*x + y*y <= 1.0);
    }
    atomicAdd(d_inside, count);
}

template<typename Sampler>
long double cuda_monte_carlo(unsigned long long trials, Sampler sampler) {
    unsigned long long h_inside = 0;
    unsigned long long *d_inside;
    
    CUDA_CHECK(cudaMalloc(&d_inside, sizeof(unsigned long long)));
    CUDA_CHECK(cudaMemset(d_inside, 0, sizeof(unsigned long long)));
    
    const int blocks = 1024, threads = 256;
    sampler(trials, blocks, threads, d_inside);
    
    CUDA_CHECK(cudaMemcpy(&h_inside, d_inside, sizeof(unsigned long long), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_inside));
    
    return 4.0L * h_inside / trials;
}

// ======================================================================
// Sampling Strategies
// ======================================================================
auto uniform_sampler = [](auto trials, int blocks, int threads, auto d_inside) {
    monte_carlo_kernel<<<blocks, threads>>>(trials, 0,1,0,1, d_inside);
};

auto stratified_x_sampler = [](auto trials, int blocks, int threads, auto d_inside) {
    const int strata = threads;
    trials /= strata;
    for(int s=0; s<strata; s++) {
        double x_min = s/double(strata);
        double x_max = (s+1)/double(strata);
        monte_carlo_kernel<<<blocks, threads>>>(trials, x_min, x_max, 0,1, d_inside);
    }
};

auto grid_stratified_sampler = [](auto trials, int blocks, int threads, auto d_inside) {
    const int grid_dim = std::sqrt(threads);
    trials /= (grid_dim*grid_dim);
    for(int i=0; i<grid_dim; i++) {
        for(int j=0; j<grid_dim; j++) {
            unsigned seed = i*grid_dim + j;
            monte_carlo_kernel<<<blocks, threads>>>(trials, 
                i/double(grid_dim), (i+1)/double(grid_dim),
                j/double(grid_dim), (j+1)/double(grid_dim),
                d_inside, seed
            );
        }
    }
};

// ======================================================================
// Main Controller
// ======================================================================
void run_precision_study() {
    const std::vector<int> precisions{5,10,15,20};
    const int runs = 5;
    
    for(int precision : precisions) {
        std::cout << GREEN << "Precision: " << precision << "\n\n";

        std::string req_trials_str = computeRequiredTrialsStrMC(precision);
        std::cout << BLUE << "Required trials for " << precision << " decimal places: " 
                  << req_trials_str << "\n";

        unsigned long long trials = computeRequiredTrialsMC(precision);
        std::cout << "Using " << trials << " trials.\n\n" << RESET;

        auto run = [&](auto sampler, const std::string& name) {
            auto stats = measureSimulation(runs, [=]{ return cuda_monte_carlo(trials, sampler); });
            report_results("results_precisions_mc.csv", 
                std::to_string(precision)+","+name, precision, stats);
        };

        run(uniform_sampler, "CUDA Uniform");
        run(stratified_x_sampler, "CUDA Stratified X");
        run(grid_stratified_sampler, "CUDA Grid Stratified");
    }
}

void run_scaling_study() {
    const std::vector<unsigned long long> trials{1ULL<<24, 1ULL<<26, 1ULL<<28};
    const int precision = 16, runs = 5;
    
    for(auto t : trials) {
        std::cout << GREEN << "Trials: " << t << "\n\n" << RESET;

        auto run = [&](auto sampler, const std::string& name) {
            auto stats = measureSimulation(runs, [=]{ return cuda_monte_carlo(t, sampler); });
            report_results("results_trials.csv", 
                std::to_string(t)+","+name, precision, stats);
        };

        run(uniform_sampler, "CUDA Uniform");
        run(stratified_x_sampler, "CUDA Stratified X");
        run(grid_stratified_sampler, "CUDA Grid Stratified");
    }
}

// ======================================================================
// Main Function
// ======================================================================
int main() {
    printCudaInfo();
    run_precision_study();
    run_scaling_study();
    return 0;
}
