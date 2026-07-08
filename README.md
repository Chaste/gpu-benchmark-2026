# Chaste FLAME GPU Paper 2026

The main benchmark code can be found in `/apps/src`

## Before Building

### Pre-requisites
- Chaste-compatible operating system
- NVIDIA GPU
- NVIDIA CUDA toolkit installed

### Setting up Chaste

1. Set up chaste according to [its instructions](https://chaste.github.io/docs/installguides/).
2. Switch to the branch containing the GPU-aware classes `git checkout flamegpu-3d-integration`

### Setting up the Benchmark Repository

1. Clone this repository into your chaste projects directory

```
cd projects
git clone https://github.com/Chaste/gpu-benchmark-2026.git
```

## Build Instructions

1. Create a build directory outside of the chaste source tree

```
mkdir build
```

2. Configure chaste to build with CUDA support and to build the `gpu-benchmark-2026` project.

```
cmake ../Chaste -DChaste_ENABLE_project_gpu-benchmark-2026=ON -DChaste_ENABLE_project_gpu-benchmark-2026_APPS=ON --compile-no-warning-as-error -DCMAKE_CUDA_ARCHITECTURES=61 -DCMAKE_BUILD_TYPE=Release -DFLAMEGPU_SEATBELTS=OFF
```

The `--compile-no-warning-as-error` flag is used as modern compilers issue new warnings which have not been fixed in the pinned FLAME GPU version.

The `CMAKE_CUDA_ARCHITECTURES` flag should be set to a model that is compatible with your GPU and CUDA version.

3. Build the project
```
make -jN project_gpu-benchmark-2026
```
where `N` is replaced by the number of threads you want to use to build.

## Running the Benchmark

The benchmark is run by calling the created executable. From the build directory, run

```
./projects/gpu-benchmark-2026/apps/ExampleApp_gpu-benchmark-2026.cu
```

## Generating Graphs

Navigate to the `chaste-flamegpu-paper-results` directory

Run:

```
python3 generate-graphs.py
```

This uses the data produced for the paper to generate the relevant figures. The data used by the script is processed data. For simulations which were repeated multiple times, only the aggregate data (means and errors) was retained. Errors are either stored in a column in the results csv files or in separate csv files depending on the specific dataset.
