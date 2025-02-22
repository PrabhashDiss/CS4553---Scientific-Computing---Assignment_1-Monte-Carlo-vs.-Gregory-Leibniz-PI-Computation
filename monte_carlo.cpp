// monte_carlo.cpp
#include <iostream>
#include <pthread.h>
#include <random>
#include <tuple>
#include <vector>
#include "monte_carlo.h"

#ifdef _OPENMP
  #include <omp.h>
#endif

unsigned long long count_circle_hits(unsigned long long trials,
                               double x_min, double x_max,
                               double y_min, double y_max,
                               unsigned long long seed_offset = 0) {
    unsigned long long hits = 0;
    std::mt19937_64 rng(std::random_device{}() + seed_offset);
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    for (unsigned long long i = 0; i < trials; i++) {
        double x = x_min + (x_max - x_min) * dist(rng);
        double y = y_min + (y_max - y_min) * dist(rng);
        if (x * x + y * y <= 1.0)
            hits++;
    }
    return hits;
}

// Thread worker function
void* monte_carlo_worker(void* args) {
    auto* params = static_cast<std::tuple<unsigned long long*, unsigned long long, double, double, double, double, unsigned long long>*>(args);
    auto [result_ptr, trials, x_min, x_max, y_min, y_max, seed_offset] = *params;
    delete params;

    unsigned long long hits = count_circle_hits(trials, x_min, x_max, y_min, y_max, seed_offset);
    *result_ptr = hits;

    pthread_exit(nullptr);
}

// Single-threaded with uniform sampling
mp_decimal_float monte_carlo_pi_singlethreaded_with_uniform(unsigned long long trials) {
    unsigned long long total_hits = count_circle_hits(trials, 0.0, 1.0, 0.0, 1.0);
    return mp_decimal_float(4.0) * mp_decimal_float(total_hits) / mp_decimal_float(trials);
}

// Multi-threaded using pthread with uniform sampling
mp_decimal_float monte_carlo_pi_multithreaded_using_pthread_with_uniform(unsigned long long trials, unsigned int num_threads) {
    std::vector<unsigned long long> results(num_threads);
    std::vector<pthread_t> threads(num_threads);

    unsigned long long trials_per_thread = trials / num_threads;

    for (unsigned int i = 0; i < num_threads; ++i) {
        double x_min = 0, x_max = 1, y_min = 0, y_max = 1;
        unsigned long long seed_offset = i;

        auto* thread_params = new std::tuple<unsigned long long*, unsigned long long, double, double, double, double, unsigned long long>(
            &results[i], trials_per_thread, x_min, x_max, y_min, y_max, seed_offset
        );

        if (pthread_create(&threads[i], nullptr, monte_carlo_worker, thread_params) != 0) {
            std::cerr << "Error creating thread " << i << std::endl;
            exit(1);
        }
    }

    for (auto& thread : threads) {
        pthread_join(thread, nullptr);
    }

    unsigned long long total_hits = 0;
    for (const auto& hits : results) {
        total_hits += hits;
    }

    return mp_decimal_float(4.0) * mp_decimal_float(total_hits) / mp_decimal_float(trials_per_thread * num_threads);
}

#ifdef _OPENMP
// Multi-threaded using OpenMP with uniform sampling
mp_decimal_float monte_carlo_pi_multithreaded_using_omp_with_uniform(unsigned long long trials, unsigned int num_threads) {
    unsigned long long total_hits = 0;
    unsigned long long trials_per_thread = trials / num_threads;

    #pragma omp parallel num_threads(num_threads) reduction(+:total_hits)
    {
        int tid = omp_get_thread_num();
        total_hits += count_circle_hits(trials_per_thread, 0, 1, 0, 1, tid);
    }

    return mp_decimal_float(4.0) * mp_decimal_float(total_hits) / mp_decimal_float(trials_per_thread * num_threads);
}
#endif

// Multi-threaded using pthread with stratified sampling by x–coordinate
mp_decimal_float monte_carlo_pi_multithreaded_using_pthread_with_stratified_x(unsigned long long trials, int num_strata) {
    std::vector<unsigned long long> results(num_strata);
    std::vector<pthread_t> threads(num_strata);

    unsigned long long trials_per_stratum = trials / num_strata;

    for (int s = 0; s < num_strata; s++) {
        double x_min = s * 1.0 / num_strata, x_max = (s + 1) * 1.0 / num_strata, y_min = 0, y_max = 1;
        unsigned long long seed_offset = s;

        auto* thread_params = new std::tuple<unsigned long long*, unsigned long long, double, double, double, double, unsigned long long>(
            &results[s], trials_per_stratum, x_min, x_max, y_min, y_max, seed_offset
        );

        if (pthread_create(&threads[s], nullptr, monte_carlo_worker, thread_params) != 0) {
            std::cerr << "Error creating thread " << s << std::endl;
            exit(1);
        }
    }

    for (auto& thread : threads) {
        pthread_join(thread, nullptr);
    }

    unsigned long long total_hits = 0;
    for (const auto& hits : results) {
        total_hits += hits;
    }

    return mp_decimal_float(4.0) * mp_decimal_float(total_hits) / mp_decimal_float(trials_per_stratum * num_strata);
}

#ifdef _OPENMP
// Multi-threaded using OpenMP with stratified sampling by x–coordinate
mp_decimal_float monte_carlo_pi_multithreaded_using_omp_with_stratified_x(unsigned long long trials, int num_strata) {
    unsigned long long total_hits = 0;
    unsigned long long trials_per_stratum = trials / num_strata;

    #pragma omp parallel num_threads(num_strata) reduction(+:total_hits)
    {
        int s = omp_get_thread_num();
        double x_min = s * 1.0 / num_strata, x_max = (s + 1) * 1.0 / num_strata, y_min = 0, y_max = 1;
        unsigned long long seed_offset = s;
        total_hits += count_circle_hits(trials_per_stratum, x_min, x_max, y_min, y_max, seed_offset);
    }

    return mp_decimal_float(4.0) * mp_decimal_float(total_hits) / mp_decimal_float(trials_per_stratum * num_strata);
}
#endif

// Multi-threaded using pthread with grid–based stratified sampling
mp_decimal_float monte_carlo_pi_multithreaded_using_pthread_with_stratified_grid(unsigned long long trials, int grid_dim) {
    std::vector<unsigned long long> results(grid_dim * grid_dim);
    std::vector<pthread_t> threads(grid_dim * grid_dim);

    unsigned long long trials_per_task = trials / (grid_dim * grid_dim);

    for (int i = 0; i < grid_dim; i++) {
        for (int j = 0; j < grid_dim; j++) {
            double x_min = i * 1.0 / grid_dim, x_max = (i + 1) * 1.0 / grid_dim, y_min = j * 1.0 / grid_dim, y_max = (j + 1) * 1.0 / grid_dim;
            unsigned long long seed_offset = i * grid_dim + j;

            auto* thread_params = new std::tuple<unsigned long long*, unsigned long long, double, double, double, double, unsigned long long>(
                &results[i * grid_dim + j], trials_per_task, x_min, x_max, y_min, y_max, seed_offset
            );

            if (pthread_create(&threads[i * grid_dim + j], nullptr, monte_carlo_worker, thread_params) != 0) {
                std::cerr << "Error creating thread " << i * grid_dim + j << std::endl;
                exit(1);
            }
        }
    }

    for (auto& thread : threads) {
        pthread_join(thread, nullptr);
    }

    unsigned long long total_hits = 0;
    for (const auto& hits : results) {
        total_hits += hits;
    }

    return mp_decimal_float(4.0) * mp_decimal_float(total_hits) / mp_decimal_float(trials_per_task * grid_dim * grid_dim);
}

#ifdef _OPENMP
// Multi-threaded using OpenMP with grid–based stratified sampling
mp_decimal_float monte_carlo_pi_multithreaded_using_omp_with_stratified_grid(unsigned long long trials, int grid_dim) {
    unsigned long long total_hits = 0;
    unsigned long long trials_per_task = trials / (grid_dim * grid_dim);

    #pragma omp parallel num_threads(grid_dim * grid_dim) reduction(+:total_hits)
    {
        int i = omp_get_thread_num() / grid_dim;
        int j = omp_get_thread_num() % grid_dim;
        double x_min = i * 1.0 / grid_dim, x_max = (i + 1) * 1.0 / grid_dim, y_min = j * 1.0 / grid_dim, y_max = (j + 1) * 1.0 / grid_dim;
        unsigned long long seed_offset = i * grid_dim + j;
        total_hits += count_circle_hits(trials_per_task, x_min, x_max, y_min, y_max, seed_offset);
    }

    return mp_decimal_float(4.0) * mp_decimal_float(total_hits) / mp_decimal_float(trials_per_task * grid_dim * grid_dim);
}
#endif
