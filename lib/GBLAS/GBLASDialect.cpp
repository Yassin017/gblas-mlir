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

LogicalResult EWiseAddOp::verify() {
  // 1. Force Ranked Tensors (Better than dyn_cast for safety)
  auto lhsType = llvm::dyn_cast<RankedTensorType>(getLhs().getType());
  auto rhsType = llvm::dyn_cast<RankedTensorType>(getRhs().getType());
  auto resType = llvm::dyn_cast<RankedTensorType>(getResult().getType());

  if (!lhsType || !rhsType || !resType)
    return emitOpError("requires ranked tensor types for all operands and results");

  // 2. Shape Matching
  if (lhsType.getShape() != rhsType.getShape())
    return emitOpError("input shapes must match");
  if (lhsType.getShape() != resType.getShape())
    return emitOpError("result shape must match input shapes");

  // 3. Element Type Matching (Avoid adding floats to ints)
  if (lhsType.getElementType() != rhsType.getElementType())
    return emitOpError("input element types must match");

  // 4. Sparsity "Sanity Check"
  // If you want to PRIORITIZE sparse, you can check for the encoding attribute.
  auto lhsEncoding = lhsType.getEncoding();
  auto rhsEncoding = rhsType.getEncoding();
  auto resEncoding = resType.getEncoding();

  // Logic: In GraphBLAS ewise_add (Union), if inputs are sparse, 
  // the result MUST be sparse to avoid a memory explosion.
  if ((lhsEncoding || rhsEncoding) && !resEncoding) {
    return emitOpError("result must have a sparse encoding if inputs are sparse");
  }

  return success();
}