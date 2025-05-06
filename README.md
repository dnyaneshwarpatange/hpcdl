# High Performance Computing and Deep Learning Project

This repository contains various implementations of parallel computing algorithms and machine learning models using CUDA, OpenMP, and Python.

## Prerequisites

### For CUDA Programs
- NVIDIA GPU with CUDA support
- CUDA Toolkit (version 11.0 or higher)
- GCC/G++ compiler

### For OpenMP Programs
- GCC/G++ compiler with OpenMP support
- Windows: MinGW or Visual Studio
- Linux: GCC/G++

### For Python Programs
- Python 3.7 or higher
- Required Python packages:
  - numpy
  - pandas
  - scikit-learn
  - tensorflow/keras

## Installation

1. Install CUDA Toolkit:
   - Download from [NVIDIA CUDA Toolkit](https://developer.nvidia.com/cuda-downloads)
   - Follow installation instructions for your operating system

2. Install Python dependencies:
```bash
pip install numpy pandas scikit-learn tensorflow
```

## Compilation and Running Instructions

### CUDA Programs

1. Compile CUDA programs:
```bash
nvcc cuda.cu -o cuda_program
```

2. Run CUDA program:
```bash
./cuda_program
```

### OpenMP Programs

1. Compile OpenMP programs:
```bash
# For BFS/DFS
g++ -fopenmp bfs_dfs_omp.cpp -o bfs_dfs_omp

# For sorting
g++ -fopenmp sorting.cpp -o sorting

# For min-max
g++ -fopenmp min_max.cpp -o min_max
```

2. Run OpenMP programs:
```bash
# For BFS/DFS
./bfs_dfs_omp

# For sorting
./sorting

# For min-max
./min_max
```

### Python Programs

1. Run Python programs:
```bash
# For IMDB dataset
python IMDB.py

# For Boston Housing dataset
python boston.py

# For Fashion MNIST dataset
python Fashion.py
```

## Program Descriptions

### CUDA Programs
- `cuda.cu`: Contains two CUDA programs:
  1. Matrix multiplication implementation
  2. Vector addition implementation

### OpenMP Programs
- `bfs_dfs_omp.cpp`: Parallel implementation of Breadth-First Search and Depth-First Search
- `sorting.cpp`: Parallel sorting algorithm implementation
- `min_max.cpp`: Parallel implementation of finding minimum and maximum values

### Python Programs
- `IMDB.py`: Text classification on IMDB dataset
- `boston.py`: Regression analysis on Boston Housing dataset
- `Fashion.py`: Image classification on Fashion MNIST dataset

## Notes
- For CUDA programs, ensure your GPU supports CUDA and the CUDA toolkit is properly installed
- For OpenMP programs, the number of threads can be controlled using the `OMP_NUM_THREADS` environment variable
- Python programs require the respective datasets to be downloaded automatically or manually placed in the correct directory

## Troubleshooting
1. If CUDA compilation fails:
   - Verify CUDA installation: `nvcc --version`
   - Check GPU compatibility: `nvidia-smi`

2. If OpenMP compilation fails:
   - Verify OpenMP support: `g++ --version`
   - Ensure proper compiler flags are used

3. If Python programs fail:
   - Verify Python version: `python --version`
   - Check installed packages: `pip list`
   - Ensure all required datasets are available
