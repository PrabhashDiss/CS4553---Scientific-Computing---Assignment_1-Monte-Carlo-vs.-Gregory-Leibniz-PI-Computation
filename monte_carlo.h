#ifndef MONTE_CARLO_H
#define MONTE_CARLO_H

#include <boost/multiprecision/cpp_dec_float.hpp>
using mp_decimal_float = boost::multiprecision::cpp_dec_float_50;

// Single-threaded with uniform sampling
mp_decimal_float monte_carlo_pi_singlethreaded_with_uniform(unsigned long long trials);

// Multi-threaded using pthread with uniform sampling
mp_decimal_float monte_carlo_pi_multithreaded_using_pthread_with_uniform(unsigned long long trials, unsigned int num_threads);

#ifdef _OPENMP
// Multi-threaded using OpenMP with uniform sampling
mp_decimal_float monte_carlo_pi_multithreaded_using_omp_with_uniform(unsigned long long trials, unsigned int num_threads);
#endif

// Multi-threaded using pthread with stratified sampling by x–coordinate
mp_decimal_float monte_carlo_pi_multithreaded_using_pthread_with_stratified_x(unsigned long long trials, int num_strata);

#ifdef _OPENMP
// Multi-threaded using OpenMP with stratified sampling by x–coordinate
mp_decimal_float monte_carlo_pi_multithreaded_using_omp_with_stratified_x(unsigned long long trials, int num_strata);
#endif

// Multi-threaded using pthread with grid–based stratified sampling
mp_decimal_float monte_carlo_pi_multithreaded_using_pthread_with_stratified_grid(unsigned long long trials, int grid_dim);

#ifdef _OPENMP
// Multi-threaded using OpenMP with grid–based stratified sampling
mp_decimal_float monte_carlo_pi_multithreaded_using_omp_with_stratified_grid(unsigned long long trials, int grid_dim);
#endif

#endif // MONTE_CARLO_H
