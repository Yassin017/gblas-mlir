# GBLAS MLIR Dialect

#### An MLIR dialect implementing operations given in the GraphBLAS API.


### Building our gblas-opt tool

```
mkdir build
cd build

cmake -G Ninja .. \
  -DMLIR_DIR=/path/to/llvm-project/build/lib/cmake/mlir \
  -DLLVM_DIR=/path/to/llvm-project/build/lib/cmake/llvm

ninja
```

### Running gblas-opt tool to transform MLIR code from gblas dialect

`gblas-opt` will be created in build/tools directory. It is the tool to lower our custom gblas dialect Ops to linalg + sparse_tensor + scf + arith dialects.
(we have implemented --convert-gblas-to-linalg lowering pass for all our APIs)

```
./tools/gblas-opt --allow-unregistered-dialect --convert-gblas-to-linalg test/test.mlir
```
test.mlir can contain mlir code written using ops defined in gblas dialect. You can add passes as command line options for gblas-opt.

Further lowering on the output can be done with the mlir-opt tool from llvm-project.


### Running benchmarks

The script to run benchmarks is in build_and_run_bench.sh in the repo's home directory. The c++ harness for linking together our algo implementations (BFS, RandomWalk, TriangleCount) in mlir with the c++ suitesparse based implementation.

First, set these environment variables to the path to your llvm-project/build/bin

```
export LLVM_BIN=~/path/to/llvm-project/build/bin
export LLVM_LIB=~/path/to/llvm-project/build/lib

```

Then run the build_and_run_bench.sh with the path to graph file (in .mtx format) as command line argument. Here we show graphs/1138_bus.mtx as an example. You will get benchmarking results on all the 3 algorithms mentioned. 

```
./build_and_run_bench.sh graphs/1138_bus.mtx

```

### Ops implemented

Our `gblas` dialect currently supports the following operations modeled after the GraphBLAS API:


| Operation | Description | Inputs | Output |
| :--- | :--- | :--- | :--- |
| **`gblas.from_coo`** | Builds a sparse tensor from COO format. | 2D indices, 1D values, dynamic sizes | 1D/2D Tensor |
| **`gblas.mxm`** | Matrix-matrix multiplication (`C = A * B`). | 2D Matrix A, 2D Matrix B, [Optional Mask] | 2D Matrix |
| **`gblas.mxv`** | Matrix-vector multiplication (`w = A * v`). | 2D Matrix, 1D Vector, [Optional Mask] | 1D Vector |
| **`gblas.vxm`** | Vector-matrix multiplication (`w = v * A`). | 1D Vector, 2D Matrix, 1D Out Buffer, [Optional Mask] | 1D Vector |
| **`gblas.vxv`** | Vector-vector dot product (`s = u * v`). | 1D Vector A, 1D Vector B | Scalar Tensor |
| **`gblas.ewise_add`** | Element-wise addition (set union). | Tensor A, Tensor B | Tensor |
| **`gblas.nrows`** / **`ncols`** | Returns row or column count. | Tensor | Index |
| **`gblas.update`** | In-place update of a tensor. | Source Tensor, Target Tensor, [Optional Mask] | Updated Tensor |
| **`gblas.intersect`** | Element-wise set intersection. | Tensor A, Tensor B, [Optional Mask] | Tensor |
| **`gblas.transpose`** | Transposes a matrix. | 2D Matrix | Transposed 2D Matrix |
| **`gblas.apply`** | Applies a unary/binary operator. | Tensor, [Optional Scalar] | Tensor |
| **`gblas.reduce_to_vector`**| Reduces a matrix along a given axis. | 2D Matrix | 1D Vector |
| **`gblas.reduce_to_scalar`**| Reduces a tensor down to a single value. | Tensor | Scalar |
| **`gblas.to_ptr`** / **`from_ptr`**| Casts between tensors and opaque pointers. | Tensor / Opaque Pointer | Opaque Pointer / Tensor |

( Most algebraic operations also accept `combine` and `reduce` operators (semi-rings for matrix multiply), as well as `mask_complement` flags)