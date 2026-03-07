add_mlir_dialect_library(GBLASDialect
  GBLAS/GBLASDialect.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/GBLAS

  DEPENDS
  GBLASIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRInferTypeOpInterface
  MLIRSideEffectInterfaces
)
