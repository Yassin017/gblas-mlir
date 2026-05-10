# #!/bin/bash
# set -e # Exit immediately if a command fails

# #adjust according to llvm-project path in system
# export LLVM_BIN=~/mlir_workspace/llvm-project/build/bin
# export LLVM_LIB=~/mlir_workspace/llvm-project/build/lib

# echo "[1/5] Lowering GBLAS to Linalg..."
# ./build/tools/gblas-opt ./test/bfs_benchmark.mlir --allow-unregistered-dialect --convert-gblas-to-linalg > step1_all.mlir

# echo "[2/5] Lowering to LLVM IR..."
# $LLVM_BIN/mlir-opt step1_all.mlir \
#     --pass-pipeline="builtin.module(sparsifier)" > kernel_all0.mlir

# $LLVM_BIN/mlir-opt kernel_all0.mlir \
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
# $LLVM_BIN/mlir-translate kernel_all.mlir --mlir-to-llvmir > kernel_all.ll
# llc --relocation-model=pic -filetype=obj kernel_all.ll -o kernel_all.o

# echo "[4/5] Compiling C++ Harness and Linking..."
# clang++ ./test/main_bfs_bench.cpp kernel_all.o -o test_bench_runner \
#      -L$LLVM_LIB \
#      -lmlir_c_runner_utils \
#      -lmlir_float16_utils \
#      -Wl,-rpath,$LLVM_LIB \
#      -O3  # Adding O3 to ensure the C++ overhead is minimal

# echo "[5/5] Running benchmark tests..."
# ./test_bench_runner

###############


# #!/bin/bash
# set -e # Exit immediately if a command fails

# # Adjust according to llvm-project path in system
# export LLVM_BIN=~/mlir_workspace/llvm-project/build/bin
# export LLVM_LIB=~/mlir_workspace/llvm-project/build/lib

# # Define the base names of the MLIR files we want to compile
# KERNELS=("bfs_benchmark" "randomwalk_benchmark" "tricount_benchmark")

# # Variable to hold all the generated object files for the final linking step
# ALL_OBJ_FILES=""

# echo "========================================"
# echo " Starting GBLAS Multi-Kernel Build"
# echo "========================================"

# # Loop through each kernel and apply the MLIR pipeline
# for KERNEL in "${KERNELS[@]}"; do
#     echo ">>> Processing ${KERNEL}.mlir..."

#     echo "  [1/3] Lowering GBLAS to Linalg..."
#     ./build/tools/gblas-opt ./test/${KERNEL}.mlir --allow-unregistered-dialect --convert-gblas-to-linalg > ${KERNEL}_step1.mlir

#     echo "  [2/3] Lowering to LLVM IR..."
#     $LLVM_BIN/mlir-opt ${KERNEL}_step1.mlir \
#         --pass-pipeline="builtin.module(sparsifier)" > ${KERNEL}_sparse.mlir

#     $LLVM_BIN/mlir-opt ${KERNEL}_sparse.mlir \
#         --sparse-storage-specifier-to-llvm \
#         --expand-strided-metadata \
#         --lower-affine \
#         --convert-linalg-to-loops \
#         --convert-scf-to-cf \
#         --finalize-memref-to-llvm \
#         --convert-vector-to-llvm \
#         --convert-math-to-llvm \
#         --convert-arith-to-llvm \
#         --convert-func-to-llvm \
#         --convert-cf-to-llvm \
#         --reconcile-unrealized-casts > ${KERNEL}_llvm.mlir

#     echo "  [3/3] Translating to Bitcode and Compiling Object File..."
#     $LLVM_BIN/mlir-translate ${KERNEL}_llvm.mlir --mlir-to-llvmir > ${KERNEL}.ll
#     llc --relocation-model=pic -filetype=obj ${KERNEL}.ll -o ${KERNEL}.o

#     # Append the newly created object file to our list
#     ALL_OBJ_FILES="$ALL_OBJ_FILES ${KERNEL}.o"
    
#     echo "  Successfully compiled ${KERNEL}.o"
#     echo "----------------------------------------"

#     rm -f ${KERNEL}_step1.mlir ${KERNEL}_sparse.mlir ${KERNEL}_llvm.mlir ${KERNEL}.ll
# done

# echo ">>> Compiling C++ Harness and Linking all kernels..."
# # 3. Pass ALL the object files we collected to clang++
# clang++ ./test/main_bench.cpp $ALL_OBJ_FILES -o test_bench_runner \
#      -L$LLVM_LIB \
#      -lmlir_c_runner_utils \
#      -lmlir_float16_utils \
#      -Wl,-rpath,$LLVM_LIB \
#      -O3

# echo ">>> Build complete! Running benchmark tests..."
# echo "========================================"
# ./test_bench_runner




#!/bin/bash
set -e # Exit immediately if a command fails

# Adjust according to llvm-project path in system
export LLVM_BIN=~/mlir_workspace/llvm-project/build/bin
export LLVM_LIB=~/mlir_workspace/llvm-project/build/lib

# Define the base names of the MLIR files we want to compile
KERNELS=("bfs_benchmark" "randomwalk_benchmark" "tricount_benchmark")
SS_KERNELS=("bfs_orig" "randomwalk_orig" "tricount_orig")

# Variable to hold all the generated object files for the final linking step
ALL_OBJ_FILES=""

echo "========================================"
echo " Starting GBLAS Multi-Kernel Build"
echo "========================================"

# 1. Compile SuiteSparse C++ Kernels
echo ">>> Compiling Native SuiteSparse GraphBLAS kernels..."
for SS_KERNEL in "${SS_KERNELS[@]}"; do
    clang++ -O3 -c ./test/${SS_KERNEL}.cpp -o ${SS_KERNEL}.o
    ALL_OBJ_FILES="$ALL_OBJ_FILES ${SS_KERNEL}.o"
    echo "  Successfully compiled ${SS_KERNEL}.o"
done
echo "----------------------------------------"

# 2. Compile MLIR Kernels
for KERNEL in "${KERNELS[@]}"; do
    echo ">>> Processing ${KERNEL}.mlir..."

    echo "  [1/3] Lowering GBLAS to Linalg..."
    ./build/tools/gblas-opt ./test/${KERNEL}.mlir --allow-unregistered-dialect --convert-gblas-to-linalg > ${KERNEL}_step1.mlir

    echo "  [2/3] Lowering to LLVM IR..."
    $LLVM_BIN/mlir-opt ${KERNEL}_step1.mlir \
        --pass-pipeline="builtin.module(sparsifier)" > ${KERNEL}_sparse.mlir

    $LLVM_BIN/mlir-opt ${KERNEL}_sparse.mlir \
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
        --reconcile-unrealized-casts > ${KERNEL}_llvm.mlir

    echo "  [3/3] Translating to Bitcode and Compiling Object File..."
    $LLVM_BIN/mlir-translate ${KERNEL}_llvm.mlir --mlir-to-llvmir > ${KERNEL}.ll
    llc --relocation-model=pic -filetype=obj ${KERNEL}.ll -o ${KERNEL}.o

    # Append the newly created object file to our list
    ALL_OBJ_FILES="$ALL_OBJ_FILES ${KERNEL}.o"
    
    echo "  Successfully compiled ${KERNEL}.o"
    echo "----------------------------------------"

    rm -f ${KERNEL}_step1.mlir ${KERNEL}_sparse.mlir ${KERNEL}_llvm.mlir ${KERNEL}.ll
done

# # 3. Compile C++ Harness and Link everything
# echo ">>> Compiling C++ Harness and Linking all kernels..."
# clang++ ./test/main_bench.cpp $ALL_OBJ_FILES -o test_bench_runner \
#      -L$LLVM_LIB \
#      -lmlir_c_runner_utils \
#      -lmlir_float16_utils \
#      -lgraphblas \
#      -Wl,-rpath,$LLVM_LIB \
#      -O3

# echo ">>> Build complete! Running benchmark tests..."
# echo "========================================"
# ./test_bench_runner


# 3. Compile C++ Harness and Link everything
echo ">>> Compiling C++ Harness and Linking all kernels..."
clang++ ./test/main_bench.cpp $ALL_OBJ_FILES -o test_bench_runner \
     -L$LLVM_LIB \
     -L/usr/local/lib \
     -L/opt/homebrew/lib \
     -lmlir_c_runner_utils \
     -lmlir_float16_utils \
     -lgraphblas \
     -Wl,-rpath,$LLVM_LIB \
     -Wl,-rpath,/usr/local/lib \
     -Wl,-rpath,/opt/homebrew/lib \
     -O3

echo ">>> Build complete! Running benchmark tests..."
echo "========================================"
./test_bench_runner