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

#include "mlir/Dialect/Arith/IR/Arith.h"

#include "mlir/Interfaces/SideEffectInterfaces.h"

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

Operation *GBLASDialect::materializeConstant(OpBuilder &builder, Attribute value,
                                             Type type, Location loc) {
  // Cast generic Attribute to TypedAttr so arith::ConstantOp is happy
  if (auto typedValue = llvm::dyn_cast<TypedAttr>(value)) {
    return builder.create<arith::ConstantOp>(loc, typedValue);
  }
  return nullptr;
}

// Logic for gblas.nrows
OpFoldResult NRowsOp::fold(FoldAdaptor adaptor) {
  auto type = getInput().getType();

  // 1. If it's unranked, we can't optimize. Return null.
  // The operation stays in the IR for the runtime to handle.
  auto rankedType = llvm::dyn_cast<RankedTensorType>(type);
  if (!rankedType) 
      return {}; 

  // 2. If it's ranked but dynamic (?), we can't optimize. Return null.
  // The operation stays in the IR for the runtime to handle.
  if (rankedType.isDynamicDim(0))
      return {};

  // 3. ONLY if it's ranked AND static (e.g., 500), we optimize!
  // The operation is deleted and replaced by a constant.
  return Builder(getContext()).getIndexAttr(rankedType.getDimSize(0));
}

// Fold implementation for NColsOp
OpFoldResult NColsOp::fold(FoldAdaptor adaptor) {
  auto type = getInput().getType();

  auto rankedType = llvm::dyn_cast<RankedTensorType>(type);
  if (!rankedType) 
      return {}; 

  if (rankedType.isDynamicDim(1))
      return {};

  return Builder(getContext()).getIndexAttr(rankedType.getDimSize(1));
}

LogicalResult UpdateOp::verify() {
    auto inputType = llvm::dyn_cast<RankedTensorType>(getInput().getType());
    auto outputType = llvm::dyn_cast<RankedTensorType>(getOutput().getType());

    if (!inputType || !outputType)
        return emitOpError("requires ranked tensor types");

    if (inputType.getShape() != outputType.getShape())
        return emitOpError("input and output dimensions must match");

    if (mlir::Value mask = getMask()) {
        auto maskType = llvm::dyn_cast<RankedTensorType>(mask.getType());
        if (!maskType)
            return emitOpError("mask must be a ranked tensor");
            
        if (maskType.getShape() != outputType.getShape())
            return emitOpError("mask dimensions must match output");
    }

    if (getRes().getType() != getOutput().getType())
        return emitOpError("result type must match output tensor type");
    
    return success();
}

void UpdateOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
    
    // Use '&' to pass the pointer to the operand
    effects.emplace_back(MemoryEffects::Read::get(), &getOperation()->getOpOperand(0));
    
    // Use '&' here as well for the in-place write
    effects.emplace_back(MemoryEffects::Write::get(), &getOperation()->getOpOperand(1));

    // And here for the optional mask
    if (getMask())
        effects.emplace_back(MemoryEffects::Read::get(), &getOperation()->getOpOperand(2));
}