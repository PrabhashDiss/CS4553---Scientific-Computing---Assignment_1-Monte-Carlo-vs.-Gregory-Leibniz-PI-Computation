// main.cpp
#include <algorithm>  // For sort
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>  // For accumulate
#include <random>
#include <sstream>
#include <string>
#include <thread>
#include <vector>
#include "monte_carlo.h"
#include "gregory_leibniz.h"

// Define escape codes for colors
#define RESET   "\033[0m"
#define RED     "\033[31m"
#define GREEN   "\033[32m"
#define YELLOW  "\033[33m"
#define BLUE    "\033[34m"

// ======================================================================
// Utility: Print Environment Specifications
// ======================================================================
void printEnvironmentInfo() {
    std::cout << RED << "========== Environment Specifications ==========\n";

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

// ======================================================================
// Utility: Compute Required Trials/Iterations as String
// ======================================================================
std::string computeRequiredTrialsStrMC(int precision) {
    std::ostringstream oss;
    oss << "3e" << (2 * precision);
    return oss.str();
}

std::string computeRequiredIterationsStrGL(int precision) {
    std::ostringstream oss;
    oss << "1.5e" << precision;
    return oss.str();
}

// ======================================================================
// Utility: Compute Actual Trials/Iterations
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

unsigned long long computeRequiredIterationsGL(int precision) {
    if (precision == 5) {
        return static_cast<unsigned long long>(1.5 * std::pow(10, 2));
    } else if (precision == 10) {
        return static_cast<unsigned long long>(1.5 * std::pow(10, 4));
    } else if (precision == 15) {
        return static_cast<unsigned long long>(1.5 * std::pow(10, 6));
    } else if (precision == 20) {
        return static_cast<unsigned long long>(1.5 * std::pow(10, 8));
    } else {
        return static_cast<unsigned long long>(1.5 * std::pow(10, precision - 1));  // static_cast<unsigned long long>(1.5 * std::pow(10, precision));
    }
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
    mp_decimal_float avgValue;
};

// Generic measurement function that accepts a simulation function and its arguments.
template<typename Func, typename... Args>
Stats measureSimulation(int iterations, Func simulation, Args&&... args) {
    Stats stats;
    std::vector<double> times;
    std::vector<mp_decimal_float> values;

    for (int i = 0; i < iterations; i++) {
        auto start = std::chrono::high_resolution_clock::now();
        mp_decimal_float result = mp_decimal_float(simulation(std::forward<Args>(args)...));
        auto end = std::chrono::high_resolution_clock::now();
        double elapsed = std::chrono::duration<double>(end - start).count();
        times.push_back(elapsed);
        values.push_back(result);
    }

    // Compute average value.
    stats.avgValue = mp_decimal_float(0.0);
    for (unsigned long long i = 0; i < values.size(); i++) {
        stats.avgValue += values[i];
    }
    stats.avgValue /= mp_decimal_float(values.size());
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

template<typename Func, typename... Args>
void runSimulation(const std::string& name, int iterations, std::ofstream& outFile, 
                  int precision, Func&& func, Args&&... args) {
    const int runs = 5;
    Stats stats = measureSimulation(runs, std::forward<Func>(func), std::forward<Args>(args)...);

    std::cout << name << ":\n"
              << std::fixed << std::setprecision(precision)
              << "  Average PI Value: " << stats.avgValue << "\n"
              << "  Average Time:     " << stats.avgTime << " s\n"
              << "  25th Percentile:  " << stats.p25 << " s\n"
              << "  Median Time:      " << stats.median << " s\n"
              << "  75th Percentile:  " << stats.p75 << " s\n\n";

    outFile << precision << "," << name << ","
            << std::fixed << std::setprecision(precision) << stats.avgValue << ","
            << stats.avgTime << "," << stats.p25 << "," << stats.median << "," << stats.p75 << "\n";
}

void runMonteCarloSimulations(int precision, unsigned long trials,
                             unsigned threads, std::ofstream& outFile) {
    runSimulation("Monte Carlo Single-thread with Uniform Sampling", 5, outFile, precision,
                 monte_carlo_pi_singlethreaded_with_uniform, trials);

    runSimulation("Monte Carlo Multi-thread using Pthread with Uniform Sampling", 5, outFile, precision,
                 monte_carlo_pi_multithreaded_using_pthread_with_uniform, trials, threads);

#ifdef _OPENMP
    runSimulation("Monte Carlo Multi-thread using OpenMP with Uniform Sampling", 5, outFile, precision,
                 monte_carlo_pi_multithreaded_using_omp_with_uniform, trials, threads);
#endif

    runSimulation("Monte Carlo Multi-thread using Pthread with Stratified Sampling by x-Coordinate", 5, outFile, precision,
                 monte_carlo_pi_multithreaded_using_pthread_with_stratified_x, trials, threads);

#ifdef _OPENMP
    runSimulation("Monte Carlo Multi-thread using OpenMP with Stratified Sampling by x-Coordinate", 5, outFile, precision,
                 monte_carlo_pi_multithreaded_using_omp_with_stratified_x, trials, threads);
#endif

    runSimulation("Monte Carlo Multi-thread using Pthread with Grid-based Stratified Sampling", 5, outFile, precision,
                 monte_carlo_pi_multithreaded_using_pthread_with_stratified_grid, trials, std::sqrt(threads));

#ifdef _OPENMP
    runSimulation("Monte Carlo Multi-thread using OpenMP with Grid-based Stratified Sampling", 5, outFile, precision,
                 monte_carlo_pi_multithreaded_using_omp_with_stratified_grid, trials, std::sqrt(threads));
#endif
}

void runGregoryLeibnizSimulations(int precision, unsigned long iterations,
                                  unsigned threads, std::ofstream& outFile) {
    runSimulation("Gregory-Leibniz Multi-thread using Pthread", 5, outFile, precision,
                 gregory_leibniz_pi_multithreaded_using_pthread, iterations, threads);

    runSimulation("Gregory-Leibniz Multi-thread using OpenMP", 5, outFile, precision,
                 gregory_leibniz_pi_multithreaded_using_omp, iterations, threads);
}

// ======================================================================
// Main Function
// ======================================================================
int main() {
    printEnvironmentInfo();

    std::ofstream mcPrecisions("results_precisions_mc.csv");
    std::ofstream glPrecisions("results_precisions_gl.csv");
    std::ofstream mcTrials("results_trials.csv");
    
    mcPrecisions << "Precision,Simulation Type,Average PI Value,Average Time (s),25th Percentile (s),Median Time (s),75th Percentile (s)\n";
    glPrecisions << "Precision,Simulation Type,Average PI Value,Average Time (s),25th Percentile (s),Median Time (s),75th Percentile (s)\n";
    mcTrials << "Trials,Simulation Type,Average PI Value,Average Time (s),25th Percentile (s),Median Time (s),75th Percentile (s)\n";

    const unsigned threads = 16;

    const std::vector<int> precisions = {5, 10, 15, 20};

    // Precision-based simulations
    for (int precision : precisions) {
        std::string mcTrialsStr = computeRequiredTrialsStrMC(precision);

        std::cout << GREEN << "Precision: " << precision << "\n\n";

        std::cout << BLUE << "Required trials for Monte Carlo with " << precision << " decimal places: " 
                  << mcTrialsStr << "\n";

        unsigned long long mcTrials = computeRequiredTrialsMC(precision);

        std::cout << "Using " << mcTrials << " trials.\n\n" << RESET;

        runMonteCarloSimulations(precision, mcTrials, threads, mcPrecisions);
    }

    const int trialPrecision = 16;

    const std::vector<unsigned long long> trialCounts = {(1ULL << 24), (1ULL << 26), (1ULL << 28)};

    // Trial-based simulations
    for (auto trials : trialCounts) {
        std::cout << GREEN << "\nTrials: " << trials << "\n" << RESET;
    
        runMonteCarloSimulations(trialPrecision, trials, threads, mcTrials);
    }

    // Precision-based simulations
    for (int precision : precisions) {
        std::string glIterationsStr = computeRequiredIterationsStrGL(precision);

        std::cout << GREEN << "Precision: " << precision << "\n\n";

        std::cout << BLUE << "Required iterations for Gregory-Leibniz with " << precision << " decimal places: " 
                  << glIterationsStr << "\n";

        unsigned long long glIterations = computeRequiredIterationsGL(precision);

        std::cout << "Using " << glIterations << " iterations.\n\n" << RESET;

        runGregoryLeibnizSimulations(precision, glIterations, threads, glPrecisions);
    }

    mcPrecisions.close();
    glPrecisions.close();
    mcTrials.close();

    return 0;
}
