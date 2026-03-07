#include "GBLAS/GBLASDialect.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"

using namespace mlir;
using namespace mlir::gblas;

// Include auto-generated Enums implementation
#include "GBLASEnums.cpp.inc"

// Include auto-generated Dialect implementation
#include "GBLASDialect.cpp.inc"

// Dialect Initialization
void GBLASDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "GBLASOps.cpp.inc"
      >();
}

// Include auto-generated Operations implementation
#define GET_OP_CLASSES
#include "GBLASOps.cpp.inc"
