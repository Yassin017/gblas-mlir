add_custom_target(GBLASIncGen)

set(LLVM_TARGET_DEFINITIONS GBLAS/GBLASDialect.td)
mlir_tablegen(GBLASDialect.h.inc -gen-dialect-decls)
mlir_tablegen(GBLASDialect.cpp.inc -gen-dialect-defs)
add_public_tablegen_target(GBLASDialectIncGen)
add_dependencies(GBLASIncGen GBLASDialectIncGen)

set(LLVM_TARGET_DEFINITIONS GBLAS/GBLASOps.td)
mlir_tablegen(GBLASOps.h.inc -gen-op-decls)
mlir_tablegen(GBLASOps.cpp.inc -gen-op-defs)
add_public_tablegen_target(GBLASOpsIncGen)
add_dependencies(GBLASIncGen GBLASOpsIncGen)

# Generate the Enums
set(LLVM_TARGET_DEFINITIONS GBLAS/GBLASDialect.td)
mlir_tablegen(GBLASEnums.h.inc -gen-enum-decls)
mlir_tablegen(GBLASEnums.cpp.inc -gen-enum-defs)
add_public_tablegen_target(GBLASEnumsIncGen)
add_dependencies(GBLASIncGen GBLASEnumsIncGen)

# Generate the Passes
set(LLVM_TARGET_DEFINITIONS GBLAS/GBLASPasses.td)
mlir_tablegen(GBLASPasses.h.inc -gen-pass-decls -name GBLAS)
add_public_tablegen_target(GBLASPassesIncGen)
add_dependencies(GBLASIncGen GBLASPassesIncGen)