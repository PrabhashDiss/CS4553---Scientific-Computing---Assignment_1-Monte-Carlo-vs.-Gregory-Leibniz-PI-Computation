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

// ======================================================================
// Main Function
// ======================================================================
int main() {
    printEnvironmentInfo();

    // Write results to file
    std::ofstream outFilePrecisionsMC("results_precisions_mc.csv");
    if (!outFilePrecisionsMC) {
        std::cerr << "Error opening file for writing.\n";
        return 1;
    }
    outFilePrecisionsMC << "Precision,Simulation Type,Average PI Value,Average Time (s),25th Percentile (s),Median Time (s),75th Percentile (s)\n";

    // Write results to file
    std::ofstream outFilePrecisionsGL("results_precisions_gl.csv");
    if (!outFilePrecisionsGL) {
        std::cerr << "Error opening file for writing.\n";
        return 1;
    }
    outFilePrecisionsGL << "Precision,Simulation Type,Average PI Value,Average Time (s),25th Percentile (s),Median Time (s),75th Percentile (s)\n";

    // List of target precisions: 5, 10, 15, and 20 decimal places.
    std::vector<int> precisions = {5, 10, 15, 20};

    // For demonstration we limit the simulation trials/iterations.
    const unsigned long long max_simulation_trials = 10000000ULL; // 10 million trials
    const unsigned long long max_simulation_iterations = 10000000ULL; // 10 million iterations

    unsigned int num_threads = 16;

    for (int precision : precisions) {
        std::cout << GREEN << "Precision: " << precision << "\n\n";

        std::string req_trials_str = computeRequiredTrialsStrMC(precision);
        std::cout << BLUE << "Required trials for Monte Carlo with " << precision << " decimal places: " 
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
        // Monte Carlo: Single-threaded with uniform sampling
        Stats mcSingleThreadedUniformStats = measureSimulation(iterationsCount, monte_carlo_pi_singlethreaded_with_uniform, trials);
        std::cout << "Monte Carlo Single-threaded with Uniform Sampling:\n"
              << "  Average PI Value: " << mcSingleThreadedUniformStats.avgValue << "\n"
              << "  Average Time:     " << mcSingleThreadedUniformStats.avgTime << " s\n"
              << "  25th Percentile:  " << mcSingleThreadedUniformStats.p25 << " s\n"
              << "  Median Time:      " << mcSingleThreadedUniformStats.median << " s\n"
              << "  75th Percentile:  " << mcSingleThreadedUniformStats.p75 << " s\n\n";
        outFilePrecisionsMC << precision << ",Monte Carlo Single-threaded with Uniform Sampling,"
                << std::fixed << std::setprecision(precision)
                << mcSingleThreadedUniformStats.avgValue << ","
                << mcSingleThreadedUniformStats.avgTime << ","
                << mcSingleThreadedUniformStats.p25 << ","
                << mcSingleThreadedUniformStats.median << ","
                << mcSingleThreadedUniformStats.p75 << "\n";

        // ---------------------------------------------------
        // Monte Carlo: Multi-threaded using pthread with uniform sampling
        Stats mcMultiThreadedPtreadUniformStats = measureSimulation(iterationsCount, monte_carlo_pi_multithreaded_using_pthread_with_uniform, trials, num_threads);
        std::cout << "Monte Carlo Multi-threaded using pthread with Uniform Sampling:\n"
              << "  Average PI Value: " << mcMultiThreadedPtreadUniformStats.avgValue << "\n"
              << "  Average Time:     " << mcMultiThreadedPtreadUniformStats.avgTime << " s\n"
              << "  25th Percentile:  " << mcMultiThreadedPtreadUniformStats.p25 << " s\n"
              << "  Median Time:      " << mcMultiThreadedPtreadUniformStats.median << " s\n"
              << "  75th Percentile:  " << mcMultiThreadedPtreadUniformStats.p75 << " s\n\n";
        outFilePrecisionsMC << precision << ",Monte Carlo Multi-threaded using pthread with Uniform Sampling,"
                << std::fixed << std::setprecision(precision)
                << mcMultiThreadedPtreadUniformStats.avgValue << ","
                << mcMultiThreadedPtreadUniformStats.avgTime << ","
                << mcMultiThreadedPtreadUniformStats.p25 << ","
                << mcMultiThreadedPtreadUniformStats.median << ","
                << mcMultiThreadedPtreadUniformStats.p75 << "\n";

        // ---------------------------------------------------
        // Monte Carlo: Multi-threaded using OpenMP with uniform sampling
#ifdef _OPENMP
        Stats mcMultiThreadedOMPUniformStats = measureSimulation(iterationsCount, monte_carlo_pi_multithreaded_using_omp_with_uniform, trials, num_threads);
        std::cout << "Monte Carlo Multi-threaded using OpenMP with Uniform Sampling:\n"
              << "  Average PI Value: " << mcMultiThreadedOMPUniformStats.avgValue << "\n"
              << "  Average Time:     " << mcMultiThreadedOMPUniformStats.avgTime << " s\n"
              << "  25th Percentile:  " << mcMultiThreadedOMPUniformStats.p25 << " s\n"
              << "  Median Time:      " << mcMultiThreadedOMPUniformStats.median << " s\n"
              << "  75th Percentile:  " << mcMultiThreadedOMPUniformStats.p75 << " s\n\n";
        outFilePrecisionsMC << precision << ",Monte Carlo Multi-threaded using OpenMP with Uniform Sampling,"
                << mcMultiThreadedOMPUniformStats.avgValue << ","
                << mcMultiThreadedOMPUniformStats.avgTime << ","
                << mcMultiThreadedOMPUniformStats.p25 << ","
                << mcMultiThreadedOMPUniformStats.median << ","
                << mcMultiThreadedOMPUniformStats.p75 << "\n";
#endif

        // ---------------------------------------------------
        // Monte Carlo: Multi-threaded using pthread with stratified sampling by x–coordinate
        Stats mcMultiThreadedPtreadStratifiedXStats = measureSimulation(iterationsCount, monte_carlo_pi_multithreaded_using_pthread_with_stratified_x, trials, num_threads);
        std::cout << "Monte Carlo Multi-threaded using pthread with Stratified Sampling by x-Coordinate:\n"
              << "  Average PI Value: " << mcMultiThreadedPtreadStratifiedXStats.avgValue << "\n"
              << "  Average Time:     " << mcMultiThreadedPtreadStratifiedXStats.avgTime << " s\n"
              << "  25th Percentile:  " << mcMultiThreadedPtreadStratifiedXStats.p25 << " s\n"
              << "  Median Time:      " << mcMultiThreadedPtreadStratifiedXStats.median << " s\n"
              << "  75th Percentile:  " << mcMultiThreadedPtreadStratifiedXStats.p75 << " s\n\n";
        outFilePrecisionsMC << precision << ",Monte Carlo Multi-threaded using pthread with Stratified Sampling by x-Coordinate,"
                << std::fixed << std::setprecision(precision)
                << mcMultiThreadedPtreadStratifiedXStats.avgValue << ","
                << mcMultiThreadedPtreadStratifiedXStats.avgTime << ","
                << mcMultiThreadedPtreadStratifiedXStats.p25 << ","
                << mcMultiThreadedPtreadStratifiedXStats.median << ","
                << mcMultiThreadedPtreadStratifiedXStats.p75 << "\n";

        // ---------------------------------------------------
        // Monte Carlo: Multi-threaded using OpenMP with stratified sampling by x–coordinate
#ifdef _OPENMP
        Stats mcMultiThreadedOMPStratifiedXStats = measureSimulation(iterationsCount, monte_carlo_pi_multithreaded_using_omp_with_stratified_x, trials, num_threads);
        std::cout << "Monte Carlo Multi-threaded using OpenMP with Stratified Sampling by x-Coordinate:\n"
              << "  Average PI Value: " << mcMultiThreadedOMPStratifiedXStats.avgValue << "\n"
              << "  Average Time:     " << mcMultiThreadedOMPStratifiedXStats.avgTime << " s\n"
              << "  25th Percentile:  " << mcMultiThreadedOMPStratifiedXStats.p25 << " s\n"
              << "  Median Time:      " << mcMultiThreadedOMPStratifiedXStats.median << " s\n"
              << "  75th Percentile:  " << mcMultiThreadedOMPStratifiedXStats.p75 << " s\n\n";
        outFilePrecisionsMC << precision << ",Monte Carlo Multi-threaded using OpenMP with Stratified Sampling by x-Coordinate,"
                << mcMultiThreadedOMPStratifiedXStats.avgValue << ","
                << mcMultiThreadedOMPStratifiedXStats.avgTime << ","
                << mcMultiThreadedOMPStratifiedXStats.p25 << ","
                << mcMultiThreadedOMPStratifiedXStats.median << ","
                << mcMultiThreadedOMPStratifiedXStats.p75 << "\n";
#endif

        // ---------------------------------------------------
        // Monte Carlo: Multi-threaded using pthread with grid–based stratified sampling
        Stats mcMultiThreadedPtreadStratifiedGridStats = measureSimulation(iterationsCount, monte_carlo_pi_multithreaded_using_pthread_with_stratified_grid, trials, sqrt(num_threads));
        std::cout << "Monte Carlo Multi-threaded using pthread with Grid-based Stratified Sampling:\n"
              << "  Average PI Value: " << mcMultiThreadedPtreadStratifiedGridStats.avgValue << "\n"
              << "  Average Time:     " << mcMultiThreadedPtreadStratifiedGridStats.avgTime << " s\n"
              << "  25th Percentile:  " << mcMultiThreadedPtreadStratifiedGridStats.p25 << " s\n"
              << "  Median Time:      " << mcMultiThreadedPtreadStratifiedGridStats.median << " s\n"
              << "  75th Percentile:  " << mcMultiThreadedPtreadStratifiedGridStats.p75 << " s\n\n";
        outFilePrecisionsMC << precision << ",Monte Carlo Multi-threaded using pthread with Grid-based Stratified Sampling,"
                << std::fixed << std::setprecision(precision)
                << mcMultiThreadedPtreadStratifiedGridStats.avgValue << ","
                << mcMultiThreadedPtreadStratifiedGridStats.avgTime << ","
                << mcMultiThreadedPtreadStratifiedGridStats.p25 << ","
                << mcMultiThreadedPtreadStratifiedGridStats.median << ","
                << mcMultiThreadedPtreadStratifiedGridStats.p75 << "\n";

        // ---------------------------------------------------
        // Monte Carlo: Multi-threaded using OpenMP with grid–based stratified sampling
#ifdef _OPENMP
        Stats mcMultiThreadedOMPStratifiedGridStats = measureSimulation(iterationsCount, monte_carlo_pi_multithreaded_using_omp_with_stratified_grid, trials, sqrt(num_threads));
        std::cout << "Monte Carlo Multi-threaded using OpenMP with Grid-based Stratified Sampling:\n"
              << "  Average PI Value: " << mcMultiThreadedOMPStratifiedGridStats.avgValue << "\n"
              << "  Average Time:     " << mcMultiThreadedOMPStratifiedGridStats.avgTime << " s\n"
              << "  25th Percentile:  " << mcMultiThreadedOMPStratifiedGridStats.p25 << " s\n"
              << "  Median Time:      " << mcMultiThreadedOMPStratifiedGridStats.median << " s\n"
              << "  75th Percentile:  " << mcMultiThreadedOMPStratifiedGridStats.p75 << " s\n\n";
        outFilePrecisionsMC << precision << ",Monte Carlo Multi-threaded using OpenMP with Grid-based Stratified Sampling,"
                << mcMultiThreadedOMPStratifiedGridStats.avgValue << ","
                << mcMultiThreadedOMPStratifiedGridStats.avgTime << ","
                << mcMultiThreadedOMPStratifiedGridStats.p25 << ","
                << mcMultiThreadedOMPStratifiedGridStats.median << ","
                << mcMultiThreadedOMPStratifiedGridStats.p75 << "\n";
#endif

        std::string req_iterations_str = computeRequiredIterationsStrGL(precision);
        std::cout << BLUE << "Required iterations for Gregory-Leibniz with " << precision << " decimal places: " 
                  << req_iterations_str << "\n";
        std::cout << "Decimal precision set to: " << precision << " places\n";

        // // For simulation, we use the maximum allowed iterations (for demo purposes)
        // unsigned long long iterations = max_simulation_iterations;
        // std::cout << "Using " << iterations << " iterations for simulation.\n\n";
        // Use actual iterations computed from precision instead of a fixed maximum.
        unsigned long long iterations = computeRequiredIterationsGL(precision);
        std::cout << "Using " << iterations << " iterations for simulation.\n\n";

        // Set output precision for simulation results
        std::cout << RESET << std::fixed << std::setprecision(precision);

        // ---------------------------------------------------
        // Gregory-Leibniz Series: Multi-threaded using pthread
        Stats glMultiThreadedPthreadStats = measureSimulation(iterationsCount, gregory_leibniz_pi_multithreaded_using_pthread, iterations, num_threads);
        std::cout << "Gregory-Leibniz Multi-threaded using pthread:\n"
              << "  Average PI Value: " << glMultiThreadedPthreadStats.avgValue << "\n"
              << "  Average Time:     " << glMultiThreadedPthreadStats.avgTime << " s\n"
              << "  25th Percentile:  " << glMultiThreadedPthreadStats.p25 << " s\n"
              << "  Median Time:      " << glMultiThreadedPthreadStats.median << " s\n"
              << "  75th Percentile:  " << glMultiThreadedPthreadStats.p75 << " s\n\n";
        outFilePrecisionsGL << precision << ",Gregory-Leibniz Multi-threaded using pthread,"
                << glMultiThreadedPthreadStats.avgValue << ","
                << glMultiThreadedPthreadStats.avgTime << ","
                << glMultiThreadedPthreadStats.p25 << ","
                << glMultiThreadedPthreadStats.median << ","
                << glMultiThreadedPthreadStats.p75 << "\n";

        // ---------------------------------------------------
        // Gregory-Leibniz Series: Multi-threaded using OpenMP
        Stats glMultiThreadedOMPStats = measureSimulation(iterationsCount, gregory_leibniz_pi_multithreaded_using_omp, iterations, num_threads);
        std::cout << "Gregory-Leibniz Multi-threaded using OpenMP:\n"
              << "  Average PI Value: " << glMultiThreadedOMPStats.avgValue << "\n"
              << "  Average Time:     " << glMultiThreadedOMPStats.avgTime << " s\n"
              << "  25th Percentile:  " << glMultiThreadedOMPStats.p25 << " s\n"
              << "  Median Time:      " << glMultiThreadedOMPStats.median << " s\n"
              << "  75th Percentile:  " << glMultiThreadedOMPStats.p75 << " s\n\n";
        outFilePrecisionsGL << precision << ",Gregory-Leibniz Multi-threaded using OpenMP,"
                << glMultiThreadedOMPStats.avgValue << ","
                << glMultiThreadedOMPStats.avgTime << ","
                << glMultiThreadedOMPStats.p25 << ","
                << glMultiThreadedOMPStats.median << ","
                << glMultiThreadedOMPStats.p75 << "\n";

        std::cout << "------------------------------------------------------\n\n";
    }

    outFilePrecisionsMC.close();

    outFilePrecisionsGL.close();

    // Write results to file
    std::ofstream outFileTrials("results_trials.csv");
    if (!outFileTrials) {
        std::cerr << "Error opening file for writing.\n";
        return 1;
    }
    outFileTrials << "Trials,Simulation Type,Average PI Value,Average Time (s),25th Percentile (s),Median Time (s),75th Percentile (s)\n";

    // Define the fixed trial counts.
    std::vector<unsigned long long> trialCounts = { (1ULL << 24), (1ULL << 26), (1ULL << 28) };

    int precision = 16;

    for (unsigned long long trials : trialCounts) {
        std::cout << GREEN << "Trials: " << trials << "\n\n";

        const int iterationsCount = 5;

        // Set output precision for simulation results
        std::cout << RESET << std::fixed << std::setprecision(precision);

        // ---------------------------------------------------
        // Monte Carlo: Single-threaded with uniform sampling
        Stats mcSingleThreadedUniformStats = measureSimulation(iterationsCount, monte_carlo_pi_singlethreaded_with_uniform, trials);
        std::cout << "Monte Carlo Single-threaded with Uniform Sampling:\n"
              << "  Average PI Value: " << mcSingleThreadedUniformStats.avgValue << "\n"
              << "  Average Time:     " << mcSingleThreadedUniformStats.avgTime << " s\n"
              << "  25th Percentile:  " << mcSingleThreadedUniformStats.p25 << " s\n"
              << "  Median Time:      " << mcSingleThreadedUniformStats.median << " s\n"
              << "  75th Percentile:  " << mcSingleThreadedUniformStats.p75 << " s\n\n";
        outFileTrials << trials << ",Monte Carlo Single-threaded with Uniform Sampling,"
                << std::fixed << std::setprecision(precision)
                << mcSingleThreadedUniformStats.avgValue << ","
                << mcSingleThreadedUniformStats.avgTime << ","
                << mcSingleThreadedUniformStats.p25 << ","
                << mcSingleThreadedUniformStats.median << ","
                << mcSingleThreadedUniformStats.p75 << "\n";

        // ---------------------------------------------------
        // Monte Carlo: Multi-threaded using pthread with uniform sampling
        Stats mcMultiThreadedPtreadUniformStats = measureSimulation(iterationsCount, monte_carlo_pi_multithreaded_using_pthread_with_uniform, trials, num_threads);
        std::cout << "Monte Carlo Multi-threaded using pthread with Uniform Sampling:\n"
              << "  Average PI Value: " << mcMultiThreadedPtreadUniformStats.avgValue << "\n"
              << "  Average Time:     " << mcMultiThreadedPtreadUniformStats.avgTime << " s\n"
              << "  25th Percentile:  " << mcMultiThreadedPtreadUniformStats.p25 << " s\n"
              << "  Median Time:      " << mcMultiThreadedPtreadUniformStats.median << " s\n"
              << "  75th Percentile:  " << mcMultiThreadedPtreadUniformStats.p75 << " s\n\n";
        outFileTrials << trials << ",Monte Carlo Multi-threaded using pthread with Uniform Sampling,"
                << std::fixed << std::setprecision(precision)
                << mcMultiThreadedPtreadUniformStats.avgValue << ","
                << mcMultiThreadedPtreadUniformStats.avgTime << ","
                << mcMultiThreadedPtreadUniformStats.p25 << ","
                << mcMultiThreadedPtreadUniformStats.median << ","
                << mcMultiThreadedPtreadUniformStats.p75 << "\n";

        // ---------------------------------------------------
        // Monte Carlo: Multi-threaded using OpenMP with uniform sampling
#ifdef _OPENMP
        Stats mcMultiThreadedOMPUniformStats = measureSimulation(iterationsCount, monte_carlo_pi_multithreaded_using_omp_with_uniform, trials, num_threads);
        std::cout << "Monte Carlo Multi-threaded using OpenMP with Uniform Sampling:\n"
              << "  Average PI Value: " << mcMultiThreadedOMPUniformStats.avgValue << "\n"
              << "  Average Time:     " << mcMultiThreadedOMPUniformStats.avgTime << " s\n"
              << "  25th Percentile:  " << mcMultiThreadedOMPUniformStats.p25 << " s\n"
              << "  Median Time:      " << mcMultiThreadedOMPUniformStats.median << " s\n"
              << "  75th Percentile:  " << mcMultiThreadedOMPUniformStats.p75 << " s\n\n";
        outFileTrials << trials << ",Monte Carlo Multi-threaded using OpenMP with Uniform Sampling,"
                << mcMultiThreadedOMPUniformStats.avgValue << ","
                << mcMultiThreadedOMPUniformStats.avgTime << ","
                << mcMultiThreadedOMPUniformStats.p25 << ","
                << mcMultiThreadedOMPUniformStats.median << ","
                << mcMultiThreadedOMPUniformStats.p75 << "\n";
#endif

        // ---------------------------------------------------
        // Monte Carlo: Multi-threaded using pthread with stratified sampling by x–coordinate
        Stats mcMultiThreadedPtreadStratifiedXStats = measureSimulation(iterationsCount, monte_carlo_pi_multithreaded_using_pthread_with_stratified_x, trials, num_threads);
        std::cout << "Monte Carlo Multi-threaded using pthread with Stratified Sampling by x-Coordinate:\n"
              << "  Average PI Value: " << mcMultiThreadedPtreadStratifiedXStats.avgValue << "\n"
              << "  Average Time:     " << mcMultiThreadedPtreadStratifiedXStats.avgTime << " s\n"
              << "  25th Percentile:  " << mcMultiThreadedPtreadStratifiedXStats.p25 << " s\n"
              << "  Median Time:      " << mcMultiThreadedPtreadStratifiedXStats.median << " s\n"
              << "  75th Percentile:  " << mcMultiThreadedPtreadStratifiedXStats.p75 << " s\n\n";
        outFileTrials << trials << ",Monte Carlo Multi-threaded using pthread with Stratified Sampling by x-Coordinate,"
                << std::fixed << std::setprecision(precision)
                << mcMultiThreadedPtreadStratifiedXStats.avgValue << ","
                << mcMultiThreadedPtreadStratifiedXStats.avgTime << ","
                << mcMultiThreadedPtreadStratifiedXStats.p25 << ","
                << mcMultiThreadedPtreadStratifiedXStats.median << ","
                << mcMultiThreadedPtreadStratifiedXStats.p75 << "\n";

        // ---------------------------------------------------
        // Monte Carlo: Multi-threaded using OpenMP with stratified sampling by x–coordinate
#ifdef _OPENMP
        Stats mcMultiThreadedOMPStratifiedXStats = measureSimulation(iterationsCount, monte_carlo_pi_multithreaded_using_omp_with_stratified_x, trials, num_threads);
        std::cout << "Monte Carlo Multi-threaded using OpenMP with Stratified Sampling by x-Coordinate:\n"
              << "  Average PI Value: " << mcMultiThreadedOMPStratifiedXStats.avgValue << "\n"
              << "  Average Time:     " << mcMultiThreadedOMPStratifiedXStats.avgTime << " s\n"
              << "  25th Percentile:  " << mcMultiThreadedOMPStratifiedXStats.p25 << " s\n"
              << "  Median Time:      " << mcMultiThreadedOMPStratifiedXStats.median << " s\n"
              << "  75th Percentile:  " << mcMultiThreadedOMPStratifiedXStats.p75 << " s\n\n";
        outFileTrials << trials << ",Monte Carlo Multi-threaded using OpenMP with Stratified Sampling by x-Coordinate,"
                << mcMultiThreadedOMPStratifiedXStats.avgValue << ","
                << mcMultiThreadedOMPStratifiedXStats.avgTime << ","
                << mcMultiThreadedOMPStratifiedXStats.p25 << ","
                << mcMultiThreadedOMPStratifiedXStats.median << ","
                << mcMultiThreadedOMPStratifiedXStats.p75 << "\n";
#endif

        // ---------------------------------------------------
        // Monte Carlo: Multi-threaded using pthread with grid–based stratified sampling
        Stats mcMultiThreadedPtreadStratifiedGridStats = measureSimulation(iterationsCount, monte_carlo_pi_multithreaded_using_pthread_with_stratified_grid, trials, sqrt(num_threads));
        std::cout << "Monte Carlo Multi-threaded using pthread with Grid-based Stratified Sampling:\n"
              << "  Average PI Value: " << mcMultiThreadedPtreadStratifiedGridStats.avgValue << "\n"
              << "  Average Time:     " << mcMultiThreadedPtreadStratifiedGridStats.avgTime << " s\n"
              << "  25th Percentile:  " << mcMultiThreadedPtreadStratifiedGridStats.p25 << " s\n"
              << "  Median Time:      " << mcMultiThreadedPtreadStratifiedGridStats.median << " s\n"
              << "  75th Percentile:  " << mcMultiThreadedPtreadStratifiedGridStats.p75 << " s\n\n";
        outFileTrials << trials << ",Monte Carlo Multi-threaded using pthread with Grid-based Stratified Sampling,"
                << std::fixed << std::setprecision(precision)
                << mcMultiThreadedPtreadStratifiedGridStats.avgValue << ","
                << mcMultiThreadedPtreadStratifiedGridStats.avgTime << ","
                << mcMultiThreadedPtreadStratifiedGridStats.p25 << ","
                << mcMultiThreadedPtreadStratifiedGridStats.median << ","
                << mcMultiThreadedPtreadStratifiedGridStats.p75 << "\n";

        // ---------------------------------------------------
        // Monte Carlo: Multi-threaded using OpenMP with grid–based stratified sampling
#ifdef _OPENMP
        Stats mcMultiThreadedOMPStratifiedGridStats = measureSimulation(iterationsCount, monte_carlo_pi_multithreaded_using_omp_with_stratified_grid, trials, sqrt(num_threads));
        std::cout << "Monte Carlo Multi-threaded using OpenMP with Grid-based Stratified Sampling:\n"
              << "  Average PI Value: " << mcMultiThreadedOMPStratifiedGridStats.avgValue << "\n"
              << "  Average Time:     " << mcMultiThreadedOMPStratifiedGridStats.avgTime << " s\n"
              << "  25th Percentile:  " << mcMultiThreadedOMPStratifiedGridStats.p25 << " s\n"
              << "  Median Time:      " << mcMultiThreadedOMPStratifiedGridStats.median << " s\n"
              << "  75th Percentile:  " << mcMultiThreadedOMPStratifiedGridStats.p75 << " s\n\n";
        outFileTrials << trials << ",Monte Carlo Multi-threaded using OpenMP with Grid-based Stratified Sampling,"
                << mcMultiThreadedOMPStratifiedGridStats.avgValue << ","
                << mcMultiThreadedOMPStratifiedGridStats.avgTime << ","
                << mcMultiThreadedOMPStratifiedGridStats.p25 << ","
                << mcMultiThreadedOMPStratifiedGridStats.median << ","
                << mcMultiThreadedOMPStratifiedGridStats.p75 << "\n";
#endif

        std::cout << "------------------------------------------------------\n\n";
    }

    outFileTrials.close();

    return 0;
}
