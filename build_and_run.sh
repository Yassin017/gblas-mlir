# #!/bin/bash
# set -e # Exit immediately if a command fails

# export LLVM_BIN=~/mlir_workspace/llvm-project/build/bin
# export LLVM_LIB=~/mlir_workspace/llvm-project/build/lib

# echo "[1/5] Lowering GBLAS to Linalg..."
# ./build/tools/gblas-opt ./test/bfs_standalone.mlir --allow-unregistered-dialect --convert-gblas-to-linalg > step1_all.mlir

# echo "[2/5] Lowering to LLVM IR..."
# # ~/mlir_workspace/llvm-project/build/bin/mlir-opt step1_all.mlir \
# #     --sparsification-and-bufferization \
# #     --sparse-storage-specifier-to-llvm \
# #     --expand-realloc \
# #     --convert-linalg-to-loops \
# #     --convert-vector-to-llvm \
# #     --convert-scf-to-cf \
# #     --expand-strided-metadata \
# #     --lower-affine \
# #     --finalize-memref-to-llvm \
# #     --convert-cf-to-llvm \
# #     --convert-arith-to-llvm \
# #     --convert-func-to-llvm \
# #     --reconcile-unrealized-casts > kernel_all.mlir

# #####

# # ~/mlir_workspace/llvm-project/build/bin/mlir-opt step1_all.mlir \
# #     --pass-pipeline="builtin.module(sparsifier)" \
# #     --convert-vector-to-llvm \
# #     --convert-scf-to-cf \
# #     --convert-cf-to-llvm \
# #     --convert-arith-to-llvm \
# #     --convert-func-to-llvm \
# #     --reconcile-unrealized-casts > kernel_all.mlir

# ####

# # ~/mlir_workspace/llvm-project/build/bin/mlir-opt step1_all.mlir \
# #     --pass-pipeline="builtin.module(sparsifier)" > kernel_all0.mlir

# # ~/mlir_workspace/llvm-project/build/bin/mlir-opt kernel_all0.mlir \
# #     --convert-vector-to-llvm \
# #     --convert-scf-to-cf \
# #     --convert-cf-to-llvm \
# #     --convert-arith-to-llvm \
# #     --convert-func-to-llvm \
# #     --reconcile-unrealized-casts > kernel_all.mlir


# ####


# # ~/mlir_workspace/llvm-project/build/bin/mlir-opt step1_all.mlir \
# #     --pass-pipeline="builtin.module(sparsifier)" > kernel_all0.mlir

# # ~/mlir_workspace/llvm-project/build/bin/mlir-opt kernel_all0.mlir \
# #     --sparse-storage-specifier-to-llvm \
# #     --expand-strided-metadata \
# #     --lower-affine \
# #     --convert-linalg-to-loops \
# #     --convert-scf-to-cf \
# #     --finalize-memref-to-llvm \
# #     --convert-vector-to-llvm \
# #     --convert-math-to-llvm \
# #     --convert-arith-to-llvm \
# #     --convert-func-to-llvm \
# #     --convert-cf-to-llvm \
# #     --reconcile-unrealized-casts > kernel_all.mlir

# ~/mlir_workspace/llvm-project/build/bin/mlir-opt step1_all.mlir \
#     --pass-pipeline="builtin.module(canonicalize,cse,sparsifier)" > kernel_all0.mlir

# # ~/mlir_workspace/llvm-project/build/bin/mlir-opt step1_all.mlir \
# #     --pass-pipeline="builtin.module(sparsifier)" > kernel_all0.mlir

# ~/mlir_workspace/llvm-project/build/bin/mlir-opt kernel_all0.mlir \
#     --sparse-storage-specifier-to-llvm \
#     --expand-strided-metadata \
#     --lower-affine \
#     --convert-linalg-to-loops \
#     --convert-scf-to-cf \
#     --finalize-memref-to-llvm \
#     --convert-vector-to-llvm \
#     --convert-math-to-llvm \
#     --convert-arith-to-llvm \
#     --convert-func-to-llvm \
#     --convert-cf-to-llvm \
#     --reconcile-unrealized-casts > kernel_all.mlir


# echo "[3/5] Translating to Bitcode and Compiling Object File..."
# ~/mlir_workspace/llvm-project/build/bin/mlir-translate kernel_all.mlir --mlir-to-llvmir > kernel_all.ll
# llc --relocation-model=pic -filetype=obj kernel_all.ll -o kernel_all.o

# echo "[4/5] Linking with C++..."
# # clang++ main_all_ops.cpp kernel_all.o -o test_all_runner \
# #     -L/home/hp/mlir_workspace/llvm-project/build/lib \
# #     -lmlir_c_runner_utils \
# #     -lmlir_float16_utils \
# #     -Wl,-rpath,/home/hp/mlir_workspace/llvm-project/build/lib

# # clang++ kernel_all.o -o test_all_runner \
# #      -L$LLVM_LIB -lmlir_c_runner_utils -lmlir_float16_utils     -Wl,-rpath,$LLVM_LIB

# clang++ kernel_all.o -o test_all_runner \
#      -L$LLVM_LIB \
#      -lmlir_c_runner_utils \
#      -lmlir_float16_utils \
#      -Wl,-rpath,$LLVM_LIB

# echo "[5/5] Running tests..."
# ./test_all_runner



# ######

# # #!/bin/bash
# # set -e # Exit immediately if a command fails

# # export LLVM_BIN=~/mlir_workspace/llvm-project/build/bin
# # export LLVM_LIB=~/mlir_workspace/llvm-project/build/lib

# # echo "[1/5] Lowering GBLAS to Linalg..."
# # ./build/tools/gblas-opt -allow-unregistered-dialect ./test/bfs_standalone.mlir --convert-gblas-to-linalg > step1_all.mlir

# # echo "[2/5] Lowering to LLVM IR..."

# # # STEP 2A: Sparsification and Permissive Bufferization
# # # We explicitly call one-shot-bufferize with flags to allow buffer re-allocations 
# # # across loop boundaries using the exact syntax supported by your MLIR version.
# # ~/mlir_workspace/llvm-project/build/bin/mlir-opt step1_all.mlir \
# #     --pass-pipeline="builtin.module(sparsifier, \
# #         one-shot-bufferize{allow-return-allocs-from-loops=1 copy-before-write=1 bufferize-function-boundaries=1})" \
# #     > kernel_all0.mlir

# # # STEP 2B: Standard Lowering to LLVM
# # ~/mlir_workspace/llvm-project/build/bin/mlir-opt kernel_all0.mlir \
# #     --sparse-tensor-conversion \
# #     --sparse-storage-specifier-to-llvm \
# #     --expand-strided-metadata \
# #     --lower-affine \
# #     --convert-linalg-to-loops \
# #     --convert-scf-to-cf \
# #     --finalize-memref-to-llvm \
# #     --convert-vector-to-llvm \
# #     --convert-math-to-llvm \
# #     --convert-arith-to-llvm \
# #     --convert-func-to-llvm \
# #     --convert-cf-to-llvm \
# #     --reconcile-unrealized-casts > kernel_all.mlir


# # echo "[3/5] Translating to Bitcode and Compiling Object File..."
# # ~/mlir_workspace/llvm-project/build/bin/mlir-translate kernel_all.mlir --mlir-to-llvmir > kernel_all.ll
# # llc --relocation-model=pic -filetype=obj kernel_all.ll -o kernel_all.o

# # echo "[4/5] Linking with C++..."
# # clang++ kernel_all.o -o test_all_runner \
# #      -L$LLVM_LIB \
# #      -lmlir_c_runner_utils \
# #      -lmlir_float16_utils \
# #      -Wl,-rpath,$LLVM_LIB

# # echo "[5/5] Running tests..."
# # ./test_all_runner



# ####################################################################################################################################3



#!/bin/bash
set -e # Exit immediately if a command fails

export LLVM_BIN=~/mlir_workspace/llvm-project/build/bin
export LLVM_LIB=~/mlir_workspace/llvm-project/build/lib

echo "[1/5] Lowering GBLAS to Linalg..."
./build/tools/gblas-opt ./test/bfs_standalone3.mlir --allow-unregistered-dialect --convert-gblas-to-linalg > step1_all.mlir

echo "[2/5] Lowering to LLVM IR..."
~/mlir_workspace/llvm-project/build/bin/mlir-opt step1_all.mlir \
    --pass-pipeline="builtin.module(sparsifier)" > kernel_all0.mlir

~/mlir_workspace/llvm-project/build/bin/mlir-opt kernel_all0.mlir \
    --sparse-storage-specifier-to-llvm \
    --expand-strided-metadata \
    --lower-affine \
    --convert-linalg-to-loops \
    --convert-scf-to-cf \
    --finalize-memref-to-llvm \
    --convert-vector-to-llvm \
    --convert-math-to-llvm \
    --convert-arith-to-llvm \
    --convert-func-to-llvm \
    --convert-cf-to-llvm \
    --reconcile-unrealized-casts > kernel_all.mlir

echo "[3/5] Translating to Bitcode and Compiling Object File..."
~/mlir_workspace/llvm-project/build/bin/mlir-translate kernel_all.mlir --mlir-to-llvmir > kernel_all.ll
llc --relocation-model=pic -filetype=obj kernel_all.ll -o kernel_all.o

echo "[4/5] Linking with C++..."
# clang++ main_all_ops.cpp kernel_all.o -o test_all_runner \
#     -L/home/hp/mlir_workspace/llvm-project/build/lib \
#     -lmlir_c_runner_utils \
#     -lmlir_float16_utils \
#     -Wl,-rpath,/home/hp/mlir_workspace/llvm-project/build/lib

# clang++ kernel_all.o -o test_all_runner \
#      -L$LLVM_LIB -lmlir_c_runner_utils -lmlir_float16_utils     -Wl,-rpath,$LLVM_LIB

clang++ kernel_all.o -o test_all_runner \
     -L$LLVM_LIB \
     -lmlir_c_runner_utils \
     -lmlir_float16_utils \
     -Wl,-rpath,$LLVM_LIB

echo "[5/5] Running tests..."
./test_all_runner


