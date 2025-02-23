# Makefile

# Compiler settings
CXX = g++
CXXFLAGS = -O2 -Wall -fopenmp -std=c++17
NVCC = nvcc
NVCCFLAGS = -O2 -arch=sm_89

# Profiling settings
PROF_CXXFLAGS = $(CXXFLAGS) -pg
PROF_NVCCFLAGS = $(NVCCFLAGS) -lineinfo

# C++ sources and target (uses main.cpp)
CPP_SRC = main_cpp.cpp monte_carlo.cpp gregory_leibniz.cpp
CPP_OBJ = $(CPP_SRC:.cpp=.o)
CPP_TARGET = simulation_cpp

# CUDA source and target (uses main.cu)
CU_SRC = main_cuda.cu
CU_OBJ = $(CU_SRC:.cu=.o)
CU_TARGET = simulation_cuda

# Profile version for C++ sources and target (uses profile_cpp.cpp)
PROF_CPP_SRC = profile_cpp.cpp monte_carlo.cpp
PROF_CPP_OBJ = $(PROF_CPP_SRC:.cpp=.o)
PROF_CPP_TARGET = prof_cpp

# Profile version for CUDA source and target (uses profile_cuda.cu)
PROF_CU_SRC = profile_cuda.cu
PROF_CU_OBJ = $(PROF_CU_SRC:.cu=.o)
PROF_CU_TARGET = prof_cuda

# Plotting script
PLOT_SCRIPT = plot_results.py

.PHONY: all clean

# Regular build and run
all: $(CPP_TARGET) $(CU_TARGET)
	@echo "Running C++ simulation..."
	./$(CPP_TARGET)
	@echo "Running CUDA simulation..."
	./$(CU_TARGET)
	$(MAKE) clean
	@echo "Generating plots..."
	python $(PLOT_SCRIPT)

# Profile build and run
profile: $(PROF_CPP_TARGET) $(PROF_CU_TARGET)
	@echo "Running Linux profile simulation (gprof)..."
	./$(PROF_CPP_TARGET)
	@echo "Generating prof report..."
	gprof $(PROF_CPP_TARGET) > prof_report_cpp.txt
	@echo "gprof profile report saved to prof_report_cpp.txt"
	@echo "Running CUDA profile simulation (ncu)..."
	./$(PROF_CU_TARGET)
	@echo "Generating CUDA prof report..."
	ncu ./$(PROF_CU_TARGET) > prof_report_cuda.txt
	@echo "CUDA profile report saved to prof_report_cuda.txt"
	$(MAKE) clean
	@echo "Profiling complete."

# Link the C++ executable
$(CPP_TARGET): $(CPP_OBJ)
	$(CXX) $(CXXFLAGS) -o $@ $^

# Link the CUDA executable
$(CU_TARGET): $(CU_OBJ)
	$(NVCC) $(NVCCFLAGS) -o $@ $^

# Compile profile C++ source files
$(PROF_CPP_TARGET): $(PROF_CPP_OBJ)
	$(CXX) $(PROF_CXXFLAGS) -o $@ $^

# Compile profile CUDA source file
$(PROF_CU_TARGET): $(PROF_CU_OBJ)
	$(NVCC) $(PROF_NVCCFLAGS) -o $@ $^

# Compile C++ source files
%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Compile CUDA source file
%.o: %.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

profile_cpp.o: profile_cpp.cpp
	$(CXX) $(PROF_CXXFLAGS) -c $< -o $@

profile_cuda.o: profile_cuda.cu
	$(NVCC) $(PROF_NVCCFLAGS) -c $< -o $@

# Clean up object files and executables
clean:
	rm -f $(CPP_OBJ) $(CU_OBJ) $(CPP_TARGET) $(CU_TARGET) $(PROF_CPP_OBJ) $(PROF_CU_OBJ) $(PROF_CPP_TARGET) $(PROF_CU_TARGET) gmon.out
