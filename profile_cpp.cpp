// profile_cpp.cpp
#include <algorithm>
#include <chrono>
#include <iostream>
#include <numeric>
#include <vector>
#include "monte_carlo.h"

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
    int precision = 16;

    const unsigned long long max_simulation_trials = 10000000ULL;
    unsigned long long trials = max_simulation_trials;

    const int iterationsCount = 5;

    unsigned int num_threads = 16;

    // Set output precision for simulation results
    std::cout << std::fixed << std::setprecision(precision);

    // ---------------------------------------------------
    // Monte Carlo: Multi-threaded using pthread with stratified sampling by x–coordinate
#ifdef _OPENMP
    Stats mcMultiThreadedPthreadStratifiedXStats = measureSimulation(iterationsCount, monte_carlo_pi_multithreaded_using_pthread_with_stratified_x, trials, num_threads);
    std::cout << "Monte Carlo Multi-threaded using Pthread with Stratified Sampling by x-Coordinate:\n"
            << "  Average PI Value: " << mcMultiThreadedPthreadStratifiedXStats.avgValue << "\n"
            << "  Average Time:     " << mcMultiThreadedPthreadStratifiedXStats.avgTime << " s\n"
            << "  25th Percentile:  " << mcMultiThreadedPthreadStratifiedXStats.p25 << " s\n"
            << "  Median Time:      " << mcMultiThreadedPthreadStratifiedXStats.median << " s\n"
            << "  75th Percentile:  " << mcMultiThreadedPthreadStratifiedXStats.p75 << " s\n\n";
#endif

    std::cout << "------------------------------------------------------\n\n";

    return 0;
}
