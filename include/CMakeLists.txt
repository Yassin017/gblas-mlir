add_custom_target(GBLASIncGen)

# Generate Dialect C++ from TableGen
set(LLVM_TARGET_DEFINITIONS GBLAS/GBLASOps.td)
mlir_tablegen(GBLASOps.h.inc -gen-op-decls)
mlir_tablegen(GBLASOps.cpp.inc -gen-op-defs)
mlir_tablegen(GBLASDialect.h.inc -gen-dialect-decls -dialect=gblas)
mlir_tablegen(GBLASDialect.cpp.inc -gen-dialect-defs -dialect=gblas)
mlir_tablegen(GBLASEnums.h.inc -gen-enum-decls)
mlir_tablegen(GBLASEnums.cpp.inc -gen-enum-defs)
add_public_tablegen_target(GBLASOpsIncGen)
add_dependencies(GBLASIncGen GBLASOpsIncGen)
