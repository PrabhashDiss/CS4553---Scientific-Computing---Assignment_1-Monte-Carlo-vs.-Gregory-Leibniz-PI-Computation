// gregory_leibniz.cpp
#include <pthread.h>
#include <tuple>
#include <vector>
#include "gregory_leibniz.h"

void* gregory_leibniz_worker(void* args) {
    auto* params = static_cast<std::tuple<mp_decimal_float*, unsigned long long, unsigned long long, unsigned int>*>(args);
    
    mp_decimal_float* result = std::get<0>(*params);
    unsigned long long start = std::get<1>(*params);
    unsigned long long end = std::get<2>(*params);
    unsigned int step = std::get<3>(*params);

    delete params;

    mp_decimal_float partial = 0.0;
    for (unsigned long long n = start; n < end; n += step) {
        partial += ((n % 2 == 0) ? 1.0 : -1.0) / (2 * n + 1);
    }

    *result = partial;

    return nullptr;
}

mp_decimal_float gregory_leibniz_pi_multithreaded_using_pthread(unsigned long long iterations, unsigned int num_threads) {
    std::vector<mp_decimal_float> results(num_threads);
    std::vector<pthread_t> threads(num_threads);

    for (unsigned int i = 0; i < num_threads; i++) {
        auto* thread_params = new std::tuple<mp_decimal_float*, unsigned long long, unsigned long long, unsigned int>(
            &results[i], i, iterations, num_threads
        );

        if (pthread_create(&threads[i], nullptr, gregory_leibniz_worker, thread_params) != 0) {
            std::cerr << "Error creating thread " << i << std::endl;
            exit(1);
        }
    }

    for (auto& thread : threads) {
        pthread_join(thread, nullptr);
    }

    mp_decimal_float total = 0.0;
    for (const auto& result : results) {
        total += result;
    }

    return mp_decimal_float(4.0) * total;
}

#pragma omp declare reduction(mp_sum : mp_decimal_float : omp_out += omp_in) initializer(omp_priv = 0.0)
mp_decimal_float gregory_leibniz_pi_multithreaded_using_omp(unsigned long long iterations, unsigned int num_threads) {
    mp_decimal_float result = 0.0;
    #pragma omp parallel for num_threads(num_threads) reduction(mp_sum:result)
    for (unsigned long long i = 0; i < iterations; i++) {
        result += ((i % 2 == 0) ? 1.0 : -1.0) / (2 * i + 1);
    }
    return mp_decimal_float(4.0) * result;
}
