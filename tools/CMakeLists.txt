set(LLVM_LINK_COMPONENTS Support)

add_llvm_executable(gblas-opt
  gblas-opt/gblas-opt.cpp
)

target_link_libraries(gblas-opt PRIVATE
  GBLASDialect
  MLIRIR
  MLIROptLib
  MLIRSupport
  MLIRParser
  MLIRPass
  MLIRFuncDialect
  MLIRTensorDialect
  MLIRLinalgDialect
  MLIRArithDialect
  MLIRSCFDialect
)
