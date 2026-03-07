# GBLAS MLIR Dialect

#### An MLIR dialect implementing operations given in the GraphBLAS API.


### Build

```
mkdir build
cd build

cmake -G Ninja .. \
  -DMLIR_DIR=/path/to/llvm-project/build/lib/cmake/mlir \
  -DLLVM_DIR=/path/to/llvm-project/build/lib/cmake/llvm

ninja
```

### Run

gblas-opt will be created in build/tools directory.
```
./tools/gblas-opt test.mlir
```
(test.mlir can contain mlir code written using ops defined in gblas dialect)
