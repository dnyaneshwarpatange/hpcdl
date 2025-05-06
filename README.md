# High Performance Computing and Deep Learning Project

This repository contains various implementations of parallel computing algorithms and machine learning models using CUDA, OpenMP, and Python.

## Table of Contents
1. [Prerequisites](#prerequisites)
2. [Detailed Program Execution Guide](#detailed-program-execution-guide)
3. [Installation](#installation)
4. [Program Execution Guide](#program-execution-guide)
5. [Sample Inputs and Outputs](#sample-inputs-and-outputs)
6. [Troubleshooting](#troubleshooting)

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

## Detailed Program Execution Guide

### 1. OpenMP Programs

#### BFS/DFS Implementation (`bfs_dfs_omp.cpp`)
**Purpose**: Implements parallel Breadth-First Search and Depth-First Search algorithms for graph traversal.

**Compilation**:
```bash
g++ -fopenmp bfs_dfs_omp.cpp -o bfs_dfs_omp
```

**Execution**:
```bash
./bfs_dfs_omp
```

**Sample Input**:
```
Enter the number of vertices: 5
Enter the number of edges: 4
Enter the edges (src dest):
0 1
1 2
2 3
3 4
```

**Sample Output**:
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

**Explanation**:
- The program creates an undirected graph with the specified number of vertices and edges
- BFS traverses the graph level by level, visiting all neighbors of the current vertex before moving deeper
- DFS traverses the graph by going as deep as possible along each branch before backtracking
- Both algorithms use OpenMP for parallel processing of neighbor exploration

#### Sorting Implementation (`sorting.cpp`)
**Purpose**: Demonstrates parallel sorting algorithms and compares their performance.

**Compilation**:
```bash
g++ -fopenmp sorting.cpp -o sorting
```

**Execution**:
```bash
./sorting
```

**Sample Input**: The program generates a random array of 10,000 integers.

**Sample Output**:
```
Original vector:
[Random numbers between 0-9999]

Sequential Bubble Sort: 2.345 seconds

Parallel Merge Sort: 0.567 seconds
```

**Explanation**:
- The program implements both sequential bubble sort and parallel merge sort
- Bubble sort is used as a baseline for comparison
- Merge sort uses OpenMP sections for parallel processing of subarrays
- The output shows the time taken by each algorithm

#### Min-Max Implementation (`min_max.cpp`)
**Purpose**: Finds minimum and maximum values in an array using parallel processing.

**Compilation**:
```bash
g++ -fopenmp min_max.cpp -o min_max
```

**Execution**:
```bash
./min_max
```

**Sample Input**: The program generates a random array of integers.

**Sample Output**:
```
Array size: 1000000
Minimum value: 1
Maximum value: 999999
Time taken: 0.123 seconds
```

**Explanation**:
- Uses OpenMP parallel reduction to find min and max values
- Divides the array into chunks for parallel processing
- Combines results using reduction operations

### 2. CUDA Programs

#### Matrix Multiplication and Vector Addition (`cuda.cu`)
**Purpose**: Demonstrates CUDA parallel processing with matrix multiplication and vector addition.

**Compilation**:
```bash
nvcc cuda.cu -o cuda_program
```

**Execution**:
```bash
./cuda_program
```

**Sample Output**:
```
Matrix Multiplication Result (first 10x10 elements):
[Matrix values will be displayed here]

Vector Addition Result (first 10 elements):
[Vector values will be displayed here]
```

**Explanation**:
- Matrix multiplication uses CUDA blocks and threads for parallel computation
- Vector addition demonstrates basic CUDA parallel processing
- Both operations are optimized for GPU execution

### 3. Python Programs

#### IMDB Sentiment Analysis (`IMDB.py`)
**Purpose**: Performs sentiment analysis on IMDB movie reviews using deep learning.

**Execution**:
```bash
python IMDB.py
```

**Sample Output**:
```
Training accuracy: 89.5%
Test accuracy: 85.2%
```

**Explanation**:
- Uses Keras to build a neural network for sentiment analysis
- Processes text data using word embeddings
- Trains on IMDB movie review dataset
- Outputs accuracy metrics for both training and test sets

#### Boston Housing Price Prediction (`boston.py`)
**Purpose**: Predicts house prices using the Boston Housing dataset.

**Execution**:
```bash
python boston.py
```

**Sample Output**:
```
Mean Squared Error: 21.345
R2 Score: 0.789
```

**Explanation**:
- Uses regression models to predict house prices
- Features include various housing attributes
- Outputs performance metrics (MSE and R² score)

#### Fashion MNIST Classification (`Fashion.py`)
**Purpose**: Classifies fashion items using the Fashion MNIST dataset.

**Execution**:
```bash
python Fashion.py
```

**Sample Output**:
```
Test accuracy: 91.2%
```

**Explanation**:
- Uses convolutional neural networks for image classification
- Processes 28x28 grayscale images
- Classifies into 10 different fashion categories

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
