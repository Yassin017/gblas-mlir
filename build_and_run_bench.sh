#!/bin/bash
set -e # Exit immediately if a command fails

#adjust according to llvm-project path in system
export LLVM_BIN=~/mlir_workspace/llvm-project/build/bin
export LLVM_LIB=~/mlir_workspace/llvm-project/build/lib

echo "[1/5] Lowering GBLAS to Linalg..."
./build/tools/gblas-opt ./test/bfs_benchmark.mlir --allow-unregistered-dialect --convert-gblas-to-linalg > step1_all.mlir

echo "[2/5] Lowering to LLVM IR..."
$LLVM_BIN/mlir-opt step1_all.mlir \
    --pass-pipeline="builtin.module(sparsifier)" > kernel_all0.mlir

$LLVM_BIN/mlir-opt kernel_all0.mlir \
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
$LLVM_BIN/mlir-translate kernel_all.mlir --mlir-to-llvmir > kernel_all.ll
llc --relocation-model=pic -filetype=obj kernel_all.ll -o kernel_all.o

echo "[4/5] Compiling C++ Harness and Linking..."
clang++ ./test/main_bfs_bench.cpp kernel_all.o -o test_bench_runner \
     -L$LLVM_LIB \
     -lmlir_c_runner_utils \
     -lmlir_float16_utils \
     -Wl,-rpath,$LLVM_LIB \
     -O3  # Adding O3 to ensure the C++ overhead is minimal

echo "[5/5] Running benchmark tests..."
./test_bench_runner