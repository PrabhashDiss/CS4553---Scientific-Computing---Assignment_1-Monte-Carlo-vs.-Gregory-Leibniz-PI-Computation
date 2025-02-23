# Makefile

# Compiler settings
CXX = g++
CXXFLAGS = -O2 -Wall -fopenmp -std=c++17
NVCC = nvcc
NVCCFLAGS = -O2 -arch=sm_89

# C++ sources and target (uses main.cpp)
CPP_SRC = main.cpp monte_carlo.cpp gregory_leibniz.cpp
CPP_OBJ = $(CPP_SRC:.cpp=.o)
CPP_TARGET = simulation_cpp

# CUDA source and target (uses main.cu)
CU_SRC = main.cu
CU_OBJ = $(CU_SRC:.cu=.o)
CU_TARGET = simulation_cuda

# Plotting script
PLOT_SCRIPT = plot_results.py

.PHONY: all clean

all: $(CPP_TARGET) $(CU_TARGET)
	@echo "Running C++ simulation..."
	./$(CPP_TARGET)
	@echo "Running CUDA simulation..."
	./$(CU_TARGET)
	$(MAKE) clean
	@echo "Generating plots..."
	python $(PLOT_SCRIPT)

# Link the C++ executable
$(CPP_TARGET): $(CPP_OBJ)
	$(CXX) $(CXXFLAGS) -o $@ $^

# Link the CUDA executable
$(CU_TARGET): $(CU_OBJ)
	$(NVCC) $(NVCCFLAGS) -o $@ $^

# Compile C++ source files
%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Compile CUDA source file
%.o: %.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

# Clean up object files and executables
clean:
	rm -f $(CPP_OBJ) $(CU_OBJ) $(CPP_TARGET) $(CU_TARGET)
