# CS4553 - Scientific Computing - Assignment 1: Monte Carlo vs. Gregory-Leibniz PI Computation

## Description

This project is an assignment for the CS4553 Scientific Computing course. It compares two distinct methods for approximating the value of π (PI): the **Monte Carlo method** and the **Gregory-Leibniz series**. The project includes multiple implementations:

- **CPU-based implementations**:
  - Single-threaded Monte Carlo with uniform sampling.
  - Multi-threaded Monte Carlo using pthreads and OpenMP with uniform, stratified (by x-coordinate), and grid-based stratified sampling.
  - Multi-threaded Gregory-Leibniz using pthreads and OpenMP.
- **GPU-accelerated implementations**:
  - CUDA-based Monte Carlo with uniform, stratified (by x-coordinate), and grid-based stratified sampling.

The primary objective is to evaluate these methods based on **precision** (accuracy of the PI approximation) and **performance** (computation time), exploring the effects of parallelization and sampling strategies. The project generates detailed results, performance plots, and profiling reports to facilitate this analysis.

## Setup and Installation

Follow these steps to set up and run the project on a Linux system:

1. **Clone the repository**:
   ```bash
   git clone https://github.com/PrabhashDiss/CS4553---Scientific-Computing---Assignment_1-Monte-Carlo-vs.-Gregory-Leibniz-PI-Computation.git
   ```

2. **Update the package list**:
   ```bash
   sudo apt-get update
   ```

3. **Install required dependencies**:
   - **Boost library** (for high-precision arithmetic):
     ```bash
     sudo apt install libboost-all-dev
     ```
   - **C++ compiler**: Ensure you have a compiler supporting C++17 and OpenMP (e.g., `g++`).
   - **CUDA Toolkit**: Required for GPU simulations. Install the NVIDIA CUDA Toolkit if you have a compatible NVIDIA GPU (see [NVIDIA's official site](https://developer.nvidia.com/cuda-downloads)).
   - **Python**: Required for plotting results (ensure `matplotlib` and `pandas` are installed; e.g., `pip install matplotlib pandas`).

4. **Navigate to the project directory**:
   ```bash
   cd CS4553---Scientific-Computing---Assignment_1-Monte-Carlo-vs.-Gregory-Leibniz-PI-Computation
   ```

## Running the Simulations

The project provides two main commands to execute and analyze the simulations:

- **Build and run all simulations**:
  ```bash
  make all
  ```
  This command:
  - Compiles the C++ (`main_cpp.cpp`) and CUDA (`main_cuda.cu`) code.
  - Runs simulations for both Monte Carlo and Gregory-Leibniz methods across various precisions (5, 10, 15, 20 decimal places) and trial counts (for Monte Carlo scaling study).
  - Generates CSV files with results (`results_precisions_mc.csv`, `results_precisions_gl.csv`, `results_trials.csv`).
  - Creates performance plots using `plot_results.py` (`performance_scores_plot_precisions.png`, `performance_scores_plot_trials.png`).
  - Moves all output files to the `results/` directory.

- **Profile the code**:
  ```bash
  make profile
  ```
  This command:
  - Compiles profiling versions of the code (`profile_cpp.cpp` for C++ and `profile_cuda.cu` for CUDA).
  - Runs a specific Monte Carlo simulation (multi-threaded with stratified sampling by x-coordinate for C++, uniform sampling for CUDA) with 10 million trials.
  - Generates profiling reports using `gprof` for C++ (`prof_report_cpp.txt`) and NVIDIA Nsight Compute (`ncu`) for CUDA (`prof_report_cuda.txt`, `profile.ncu-rep`).
  - Moves profiling reports to the `results/` directory.

## Directory Structure

The repository is organized as follows:

- **`README.md`**: This file.
- **`Makefile`**: Build script for compiling and running simulations and profiling.
- **`main_cpp.cpp`**: Main entry point for CPU-based C++ simulations.
- **`main_cuda.cu`**: Main entry point for GPU-based CUDA simulations.
- **`monte_carlo.cpp` / `monte_carlo.h`**: Monte Carlo method implementations (single-threaded and multi-threaded with various sampling strategies).
- **`gregory_leibniz.cpp` / `gregory_leibniz.h`**: Gregory-Leibniz series implementations (multi-threaded with pthreads and OpenMP).
- **`plot_results.py`**: Python script to generate performance score plots from CSV results.
- **`profile_cpp.cpp`**: Source file for profiling the C++ Monte Carlo implementation.
- **`profile_cuda.cu`**: Source file for profiling the CUDA Monte Carlo implementation.
- **`results/`**: Directory containing output files:
  - `results_precisions_mc.csv`: Monte Carlo results for different precisions.
  - `results_precisions_gl.csv`: Gregory-Leibniz results for different precisions.
  - `results_trials.csv`: Monte Carlo results for different trial counts.
  - `prof_report_cpp.txt`: C++ profiling report (gprof output).
  - `prof_report_cuda.txt`: CUDA profiling summary (ncu output).
  - `profile.ncu-rep`: Detailed CUDA profiling report (Nsight Compute format).
  - `performance_scores_plot_precisions.png`: Plot of performance scores for precision-based runs.
  - `performance_scores_plot_trials.png`: Plot of performance scores for trial-based runs.

## Results

The project produces the following outputs:

- **CSV Files**:
  - Contain simulation results for different precisions and trial counts.
  - Columns include: Precision, Trials, Simulation Type, Average PI Value, Absolute Error, Average Time (s), 25th Percentile (s), Median Time (s), 75th Percentile (s).
  - Example: Monte Carlo at precision 20 with 300 million trials provides PI approximations, errors, and timing statistics.

- **Performance Plots**:
  - Generated by `plot_results.py`.
  - Bar charts showing **performance scores** (calculated as `1 / (normalized error * normalized time)`) for each simulation type, sorted by performance.
  - Two plots: one for precision-based runs (`performance_scores_plot_precisions.png`) and one for trial-based runs (`performance_scores_plot_trials.png`).

- **Profiling Reports**:
  - **`prof_report_cpp.txt`**: gprof output showing time spent in functions (e.g., random number generation dominates in Monte Carlo).
  - **`prof_report_cuda.txt`**: Summary of CUDA kernel profiling with Nsight Compute, including execution time.
  - **`profile.ncu-rep`**: Detailed CUDA profiling report for further analysis with Nsight Compute tools.

## Notes

- **Dependencies**:
  - The project uses **Boost.Multiprecision** (`cpp_dec_float_50`) for high-precision arithmetic.
  - CUDA simulations require an NVIDIA GPU and the CUDA Toolkit (tested with compute capability 8.9, adjustable in `Makefile` via `-arch=sm_89`).
  - OpenMP is optional; simulations using it are conditionally compiled if supported.

- **System Requirements**:
  - Sufficient computational resources are needed for high-precision or large-trial simulations (e.g., 300 million trials at precision 20).
  - A multi-core CPU is recommended for multi-threaded runs (default: 16 threads).

- **Profiling**:
  - C++ profiling uses `gprof` and requires the `-pg` flag during compilation.
  - CUDA profiling uses NVIDIA Nsight Compute (`ncu`), which must be installed separately.
