// main.cpp
#include <algorithm> // For sort
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <future>
#include <iomanip>
#include <iostream>
#include <numeric>  // For accumulate
#include <random>
#include <sstream>
#include <thread>
#include <vector>

// For multiprocessing (POSIX only)
#ifdef __unix__
  #include <unistd.h>
  #include <sys/types.h>
  #include <sys/wait.h>
#endif

#ifdef __CUDACC__
#include <cuda_runtime.h>
#include <curand_kernel.h>
#endif

// ======================================================================
// Utility: Print Environment Specifications
// ======================================================================
void printEnvironmentInfo() {
    std::cout << "========== Environment Specifications ==========\n";
    
    // Operating System
#ifdef _WIN32
    std::cout << "Operating System: Windows\n";
#elif __APPLE__
    std::cout << "Operating System: macOS\n";
#elif __linux__
    std::cout << "Operating System: Linux\n";
#else
    std::cout << "Operating System: Unknown\n";
#endif

    // CPU Architecture
#if defined(__x86_64__) || defined(_M_X64)
    std::cout << "CPU Architecture: x86_64\n";
#elif defined(__i386) || defined(_M_IX86)
    std::cout << "CPU Architecture: x86\n";
#elif defined(__aarch64__)
    std::cout << "CPU Architecture: ARM64\n";
#else
    std::cout << "CPU Architecture: Unknown\n";
#endif

    // Number of Cores
    unsigned int cores = std::thread::hardware_concurrency();
    std::cout << "Number of CPU cores: " << cores << "\n";

    // C++ Standard and Compiler Version
    std::cout << "C++ Standard (__cplusplus): " << __cplusplus << "\n";
#ifdef __GNUC__
    std::cout << "Compiler: GCC " << __VERSION__ << "\n";
#elif defined(_MSC_VER)
    std::cout << "Compiler: MSVC " << _MSC_VER << "\n";
#elif defined(__clang__)
    std::cout << "Compiler: Clang " << __clang_version__ << "\n";
#else
    std::cout << "Compiler: Unknown\n";
#endif

    // Current Time
    auto now = std::chrono::system_clock::now();
    std::time_t now_time = std::chrono::system_clock::to_time_t(now);
    std::cout << "Current Time: " << std::ctime(&now_time);
    
    std::cout << "===============================================\n\n";
}

#ifdef __CUDACC__
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
    std::cout << "========== CUDA Device Info ==========\n";
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
#else
void printCudaInfo() {
    std::cout << "CUDA not supported on this compiler.\n";
}
#endif

// ======================================================================
// Utility: Compute Required Trials as String
// ======================================================================
std::string computeRequiredTrialsStr(int precision) {
    // If 2*precision > 18, then 10^(2*precision) > 1e18, so we use scientific notation.
    if (2 * precision > 18) {
        std::ostringstream oss;
        oss << "3e" << (2 * precision);
        return oss.str();
    } else {
        double trials_d = 3 * std::pow(10, 2 * precision);
        unsigned long long trials_val = static_cast<unsigned long long>(std::ceil(trials_d));
        return std::to_string(trials_val);
    }
}

// ======================================================================
// Utility: Compute Actual Trials
// ======================================================================
unsigned long long computeActualTrials(int precision) {
    // Check if the computed value fits in an unsigned long long.
    // The maximum value for unsigned long long is about 1.8e19.
    // For 2*precision <= 18, it should be safe.
    if (2 * precision <= 18) {
        return static_cast<unsigned long long>(3 * std::pow(10, 2 * precision));
    } else {
        std::cerr << "Precision " << precision 
                  << " too high to compute exact trials. Using fallback value.\n";
        return 10000000ULL; // fallback value
    }
}

// ======================================================================
// Monte Carlo Pi: Single-threaded
// ======================================================================
double monte_carlo_pi_singlethreaded(unsigned long long trials) {
    unsigned long long inside_circle = 0;
    std::mt19937_64 rng(std::random_device{}());
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    
    for (unsigned long long i = 0; i < trials; i++) {
        double x = dist(rng);
        double y = dist(rng);
        if (x*x + y*y <= 1.0)
            inside_circle++;
    }
    return (4.0 * inside_circle) / trials;
}

// ======================================================================
// Monte Carlo Pi: Multi-threaded
// ======================================================================
void monte_carlo_worker(unsigned long long trials, double &result) {
    unsigned long long inside_circle = 0;
    std::mt19937_64 rng(std::random_device{}());
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    
    for (unsigned long long i = 0; i < trials; i++) {
        double x = dist(rng);
        double y = dist(rng);
        if (x*x + y*y <= 1.0)
            inside_circle++;
    }
    result = (4.0 * inside_circle) / trials;
}

double monte_carlo_pi_multithreaded(unsigned long long trials, unsigned int num_threads) {
    std::vector<double> results(num_threads, 0.0);
    std::vector<std::thread> threads;
    unsigned long long trials_per_thread = trials / num_threads;
    
    for (unsigned int i = 0; i < num_threads; i++) {
        threads.emplace_back(monte_carlo_worker, trials_per_thread, std::ref(results[i]));
    }
    for (auto &t : threads) {
        t.join();
    }
    
    double sum = 0.0;
    for (double r : results)
        sum += r;
    
    return sum / num_threads;
}

// ======================================================================
// Monte Carlo Pi: Multiprocessing (POSIX only)
// ======================================================================
#ifdef __unix__
double monte_carlo_pi_multiprocessing(unsigned long long trials, unsigned int num_processes) {
    unsigned long long trials_per_process = trials / num_processes;
    // Create an array of pipes
    int pipefds[num_processes][2];
    for (unsigned int i = 0; i < num_processes; i++) {
        if (pipe(pipefds[i]) == -1) {
            perror("pipe");
            exit(EXIT_FAILURE);
        }
    }
    
    for (unsigned int i = 0; i < num_processes; i++) {
        pid_t pid = fork();
        if (pid < 0) {
            perror("fork");
            exit(EXIT_FAILURE);
        }
        else if (pid == 0) { // Child process
            // Close unused pipe read ends
            for (unsigned int j = 0; j < num_processes; j++) {
                close(pipefds[j][0]);
                if (j != i)
                    close(pipefds[j][1]);
            }
            unsigned long long inside_circle = 0;
            // Seed using random_device and pid to reduce correlation
            std::mt19937_64 rng(std::random_device{}() ^ (std::hash<pid_t>()(getpid())));
            std::uniform_real_distribution<double> dist(0.0, 1.0);
            
            for (unsigned long long j = 0; j < trials_per_process; j++) {
                double x = dist(rng);
                double y = dist(rng);
                if (x*x + y*y <= 1.0)
                    inside_circle++;
            }
            // Write result as binary data, check return value
            ssize_t bytes_written = write(pipefds[i][1], &inside_circle, sizeof(inside_circle));
            if (bytes_written != sizeof(inside_circle)) {
                perror("write failed");
                exit(EXIT_FAILURE);
            }
            close(pipefds[i][1]);
            exit(0);
        }
        // Parent process continues to next fork.
    }
    
    // Parent: read results from pipes and wait for children.
    unsigned long long total_inside_circle = 0;
    for (unsigned int i = 0; i < num_processes; i++) {
        close(pipefds[i][1]);
        unsigned long long count = 0;
        ssize_t bytes_read = read(pipefds[i][0], &count, sizeof(count));
        if (bytes_read != sizeof(count)) {
            perror("read failed");
            exit(EXIT_FAILURE);
        }
        close(pipefds[i][0]);
        total_inside_circle += count;
    }
    for (unsigned int i = 0; i < num_processes; i++) {
        wait(nullptr);
    }
    return (4.0 * total_inside_circle) / trials;
}
#endif

#ifdef __CUDACC__
// ======================================================================
// Monte Carlo Pi: CUDA
// ======================================================================
__global__ void monte_carlo_kernel(unsigned long long trials, unsigned long long *d_inside) {
    unsigned long long tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long stride = blockDim.x * gridDim.x;
    unsigned long long count = 0;
    // Initialize cuRAND state (using a fixed seed plus the thread id)
    curandState state;
    curand_init(1234ULL, tid, 0, &state);
    for (unsigned long long i = tid; i < trials; i += stride) {
        float x = curand_uniform(&state);
        float y = curand_uniform(&state);
        if (x*x + y*y <= 1.0f)
            count++;
    }
    atomicAdd(d_inside, count);
}

double monte_carlo_pi_cuda(unsigned long long trials) {
    unsigned long long h_inside = 0;
    unsigned long long *d_inside;
    cudaMalloc(&d_inside, sizeof(unsigned long long));
    cudaMemset(d_inside, 0, sizeof(unsigned long long));
    
    int threadsPerBlock = 256;
    int blocks = 1024; // Adjust based on your GPU and trials
    monte_carlo_kernel<<<blocks, threadsPerBlock>>>(trials, d_inside);
    cudaDeviceSynchronize();
    cudaMemcpy(&h_inside, d_inside, sizeof(unsigned long long), cudaMemcpyDeviceToHost);
    cudaFree(d_inside);
    
    double pi = 4.0 * static_cast<double>(h_inside) / static_cast<double>(trials);
    return pi;
}
#else
double monte_carlo_pi_cuda(unsigned long long trials) {
    std::cout << "CUDA not supported on this compiler.\n";
    return 0.0;
}
#endif

// ======================================================================
// Gregory-Leibniz Series: Single-threaded
// ======================================================================
double gregory_leibniz_pi_singlethreaded(unsigned long long iterations) {
    double sum = 0.0;
    for (unsigned long long n = 0; n < iterations; n++) {
        sum += ((n % 2 == 0) ? 1.0 : -1.0) / (2 * n + 1);
    }
    return 4.0 * sum;
}

// ======================================================================
// Gregory-Leibniz Series: Multi-threaded
// ======================================================================
void gregory_leibniz_worker(unsigned long long start, unsigned long long end, unsigned int step, double &result) {
    double partial_sum = 0.0;
    for (unsigned long long n = start; n < end; n += step) {
        // Alternate sign based on n being even or odd
        partial_sum += ((n % 2 == 0) ? 1.0 : -1.0) / (2 * n + 1);
    }
    result = partial_sum;
}

double gregory_leibniz_pi_multithreaded(unsigned long long iterations, unsigned int num_threads) {
    std::vector<double> results(num_threads, 0.0);
    std::vector<std::thread> threads;
    
    for (unsigned int i = 0; i < num_threads; i++) {
        // Each thread starts at a different index with a stride equal to num_threads.
        threads.emplace_back(gregory_leibniz_worker, i, iterations, num_threads, std::ref(results[i]));
    }
    for (auto &t : threads)
        t.join();
    
    double total = 0.0;
    for (double r : results)
        total += r;
    
    return 4.0 * total;
}

// ======================================================================
// Gregory-Leibniz Series: Multiprocessing (POSIX only)
// ======================================================================
#ifdef __unix__
double gregory_leibniz_pi_multiprocessing(unsigned long long iterations, unsigned int num_processes) {
    // Create an array of pipes
    int pipefds[num_processes][2];
    for (unsigned int i = 0; i < num_processes; i++) {
        if (pipe(pipefds[i]) == -1) {
            perror("pipe");
            exit(EXIT_FAILURE);
        }
    }
    
    // Fork processes
    for (unsigned int i = 0; i < num_processes; i++) {
        pid_t pid = fork();
        if (pid < 0) {
            perror("fork");
            exit(EXIT_FAILURE);
        }
        else if (pid == 0) { // Child process
            // Close unused read ends
            for (unsigned int j = 0; j < num_processes; j++) {
                close(pipefds[j][0]);
                if (j != i)
                    close(pipefds[j][1]);
            }
            double partial_sum = 0.0;
            // Each child computes the series for indices starting at its process id,
            // stepping by the total number of processes.
            for (unsigned long long n = i; n < iterations; n += num_processes) {
                partial_sum += ((n % 2 == 0) ? 1.0 : -1.0) / (2 * n + 1);
            }
            ssize_t bytes_written = write(pipefds[i][1], &partial_sum, sizeof(partial_sum));
            if (bytes_written != sizeof(partial_sum)) {
                perror("write failed");
                exit(EXIT_FAILURE);
            }
            close(pipefds[i][1]);
            exit(EXIT_SUCCESS);
        }
    }
    // Parent process: aggregate results from all children.
    double total_partial = 0.0;
    for (unsigned int i = 0; i < num_processes; i++) {
        close(pipefds[i][1]);
        double partial = 0.0;
        ssize_t bytes_read = read(pipefds[i][0], &partial, sizeof(partial));
        if (bytes_read != sizeof(partial)) {
            perror("read failed");
            exit(EXIT_FAILURE);
        }
        total_partial += partial;
        close(pipefds[i][0]);
    }
    for (unsigned int i = 0; i < num_processes; i++) {
        wait(nullptr);
    }
    return 4.0 * total_partial;
}
#endif

// ======================================================================
// Main Function
// ======================================================================
int main() {
    printEnvironmentInfo();
#ifdef __CUDACC__
    printCudaInfo();
#endif

    // List of target precisions: 5, 10, 15, and 20 decimal places.
    std::vector<int> precisions = {5, 10, 15, 20};

    // For demonstration we limit the simulation trials.
    const unsigned long long max_simulation_trials = 10000000ULL; // 10 million trials

    unsigned int num_threads = 4;
#ifdef __unix__
    unsigned int num_processes = 4;
#endif
    unsigned long long iterations = 10000000ULL; // For Gregory-Leibniz series

    for (int precision : precisions) {
        std::string req_trials_str = computeRequiredTrialsStr(precision);
        std::cout << "Required trials for " << precision << " decimal places: " 
                  << req_trials_str << "\n";
        std::cout << "Decimal precision set to: " << precision << " places\n";
        
        // For simulation, we use the maximum allowed trials (for demo purposes)
        unsigned long long trials = max_simulation_trials;
        std::cout << "Using " << trials << " trials for simulation.\n";
        // // Use actual trials computed from precision instead of a fixed maximum.
        // unsigned long long trials = computeActualTrials(precision);
        // std::cout << "Using " << trials << " trials for simulation.\n";
        
        // Set output precision for simulation results
        std::cout << std::fixed << std::setprecision(precision);
        
        // ---------------------------------------------------
        // Monte Carlo: Single-threaded
        auto start = std::chrono::high_resolution_clock::now();
        double pi_single = monte_carlo_pi_singlethreaded(trials);
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - start;
        std::cout << "Monte Carlo (Single-threaded): " << pi_single 
                  << ", Time: " << elapsed.count() << " s\n";
        
        // ---------------------------------------------------
        // Monte Carlo: Multi-threaded
        start = std::chrono::high_resolution_clock::now();
        double pi_multi = monte_carlo_pi_multithreaded(trials, num_threads);
        end = std::chrono::high_resolution_clock::now();
        elapsed = end - start;
        std::cout << "Monte Carlo (Multi-threaded):  " << pi_multi 
                  << ", Time: " << elapsed.count() << " s\n";
        
#ifdef __unix__
        // ---------------------------------------------------
        // Monte Carlo: Multiprocessing
        start = std::chrono::high_resolution_clock::now();
        double pi_proc = monte_carlo_pi_multiprocessing(trials, num_processes);
        end = std::chrono::high_resolution_clock::now();
        elapsed = end - start;
        std::cout << "Monte Carlo (Multiprocessing): " << pi_proc 
                  << ", Time: " << elapsed.count() << " s\n";
#else
        std::cout << "Monte Carlo (Multiprocessing): Not supported on this OS.\n";
#endif

#ifdef __CUDACC__
        // ---------------------------------------------------
        // Monte Carlo: CUDA
        start = std::chrono::high_resolution_clock::now();
        double pi_cuda = monte_carlo_pi_cuda(trials);
        end = std::chrono::high_resolution_clock::now();
        elapsed = end - start;
        std::cout << "Monte Carlo (CUDA):            " << pi_cuda 
                  << ", Time: " << elapsed.count() << " s\n";
#else
        std::cout << "Monte Carlo (CUDA): Not supported on this compiler.\n";
#endif

        // ---------------------------------------------------
        // Gregory-Leibniz Series: Single-threaded
        start = std::chrono::high_resolution_clock::now();
        double pi_greg_single = gregory_leibniz_pi_singlethreaded(iterations);
        end = std::chrono::high_resolution_clock::now();
        elapsed = end - start;
        std::cout << "Gregory-Leibniz (Single-threaded): " << pi_greg_single 
                  << ", Time: " << elapsed.count() << " s\n";
        
        // ---------------------------------------------------
        // Gregory-Leibniz Series: Multi-threaded
        start = std::chrono::high_resolution_clock::now();
        double pi_greg = gregory_leibniz_pi_multithreaded(iterations, num_threads);
        end = std::chrono::high_resolution_clock::now();
        elapsed = end - start;
        std::cout << "Gregory-Leibniz (Multi-threaded): " << pi_greg 
                  << ", Time: " << elapsed.count() << " s\n";
        
#ifdef __unix__
        // ---------------------------------------------------
        // Gregory-Leibniz Series: Multiprocessing
        start = std::chrono::high_resolution_clock::now();
        double pi_greg_proc = gregory_leibniz_pi_multiprocessing(iterations, num_processes);
        end = std::chrono::high_resolution_clock::now();
        elapsed = end - start;
        std::cout << "Gregory-Leibniz (Multiprocessing): " << pi_greg_proc 
                  << ", Time: " << elapsed.count() << " s\n";
#else
        std::cout << "Gregory-Leibniz (Multiprocessing): Not supported on this OS.\n";
#endif

        std::cout << "------------------------------------------------------\n";
    }
    
    return 0;
}
