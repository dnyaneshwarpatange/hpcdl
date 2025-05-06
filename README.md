# High Performance Computing and Deep Learning Project

This repository contains various implementations of parallel computing algorithms and machine learning models using CUDA, OpenMP, and Python.

## Table of Contents
1. [Prerequisites](#prerequisites)
2. [Installation](#installation)
3. [Program Execution Guide](#program-execution-guide)
4. [Sample Inputs and Outputs](#sample-inputs-and-outputs)
5. [Troubleshooting](#troubleshooting)

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

## Program Execution Guide

### 1. OpenMP Programs

#### BFS/DFS Implementation
```bash
# Compile
g++ -fopenmp bfs_dfs_omp.cpp -o bfs_dfs_omp

# Run
./bfs_dfs_omp
```

#### Sorting Implementation
```bash
# Compile
g++ -fopenmp sorting.cpp -o sorting

# Run
./sorting
```

#### Min-Max Implementation
```bash
# Compile
g++ -fopenmp min_max.cpp -o min_max

# Run
./min_max
```

### 2. CUDA Programs
```bash
# Compile
nvcc cuda.cu -o cuda_program

# Run
./cuda_program
```

### 3. Python Programs
```bash
# Run IMDB dataset
python IMDB.py

# Run Boston Housing dataset
python boston.py

# Run Fashion MNIST dataset
python Fashion.py
```

## Sample Inputs and Outputs

### 1. BFS/DFS Program
Sample Input:
```
Enter the number of vertices: 5
Enter the number of edges: 4
Enter the edges (src dest):
0 1
1 2
2 3
3 4
```

Sample Output:
```
Parallel BFS:
Visited: 0
Visited: 1
Visited: 2
Visited: 3
Visited: 4

Parallel DFS:
Visited: 0
Visited: 1
Visited: 2
Visited: 3
Visited: 4
```

### 2. Sorting Program
Sample Output:
```
Original vector:
[Random numbers between 0-9999]

Sequential Bubble Sort: X.XXX seconds

Parallel Merge Sort: X.XXX seconds
```

### 3. CUDA Program
Sample Output:
```
Matrix Multiplication Result (first 10x10 elements):
[Matrix values]

Vector Addition Result (first 10 elements):
[Vector values]
```

### 4. Python Programs

#### IMDB Dataset
Sample Output:
```
Training accuracy: XX.XX%
Test accuracy: XX.XX%
```

#### Boston Housing Dataset
Sample Output:
```
Mean Squared Error: X.XXX
R2 Score: X.XXX
```

#### Fashion MNIST Dataset
Sample Output:
```
Test accuracy: XX.XX%
```

## Program Descriptions

### OpenMP Programs
- `bfs_dfs_omp.cpp`: Parallel implementation of Breadth-First Search and Depth-First Search
  - Uses OpenMP for parallel processing of graph traversal
  - Supports undirected graphs
  - Implements both BFS and DFS algorithms

- `sorting.cpp`: Parallel sorting algorithm implementation
  - Implements both sequential bubble sort and parallel merge sort
  - Compares performance between sequential and parallel implementations
  - Uses OpenMP sections for parallel processing

- `min_max.cpp`: Parallel implementation of finding minimum and maximum values
  - Uses OpenMP for parallel reduction
  - Efficiently finds min and max values in an array

### CUDA Programs
- `cuda.cu`: Contains two CUDA programs:
  1. Matrix multiplication implementation
     - Uses CUDA blocks and threads for parallel computation
     - Optimized for square matrices
  2. Vector addition implementation
     - Demonstrates basic CUDA parallel processing
     - Uses thread indexing for parallel addition

### Python Programs
- `IMDB.py`: Text classification on IMDB dataset
  - Implements sentiment analysis
  - Uses deep learning for text classification

- `boston.py`: Regression analysis on Boston Housing dataset
  - Predicts house prices
  - Uses machine learning regression models

- `Fashion.py`: Image classification on Fashion MNIST dataset
  - Classifies fashion items
  - Uses convolutional neural networks

## Troubleshooting

### 1. CUDA Issues
- Verify CUDA installation: `nvcc --version`
- Check GPU compatibility: `nvidia-smi`
- Ensure proper CUDA toolkit version is installed

### 2. OpenMP Issues
- Verify OpenMP support: `g++ --version`
- Check compiler flags: `-fopenmp` must be included
- Set number of threads: `export OMP_NUM_THREADS=4`

### 3. Python Issues
- Verify Python version: `python --version`
- Check installed packages: `pip list`
- Ensure datasets are available in correct directories
- Check for GPU support in TensorFlow if using GPU acceleration
