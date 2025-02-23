#ifndef GREGORY_LEIBNIZ_H
#define GREGORY_LEIBNIZ_H

#include <boost/multiprecision/cpp_dec_float.hpp>
using mp_decimal_float = boost::multiprecision::cpp_dec_float_50;

mp_decimal_float gregory_leibniz_pi_multithreaded_using_pthread(unsigned long long iterations, unsigned int num_threads);
mp_decimal_float gregory_leibniz_pi_multithreaded_using_omp(unsigned long long iterations, unsigned int num_threads);

#endif // GREGORY_LEIBNIZ_H
