#include "GBLAS/GBLASDialect.h"
#include "GBLAS/GBLASPasses.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Dialect/SparseTensor/IR/SparseTensor.h"
#include "mlir/AsmParser/AsmParser.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"

#include "mlir/Dialect/Math/IR/Math.h"

#include "mlir/IR/BuiltinTypes.h"

// -- Generate Pass Implementation 
namespace mlir {
namespace gblas {
#define GEN_PASS_DEF_CONVERTGBLASTOLINALG
#include "GBLASPasses.h.inc"
} // namespace gblas
} // namespace mlir
// --

using namespace mlir;

namespace {

// Common helper to get the initialized output tensor (Empty for Sparse, Zero-Filled for Dense)
static Value getInitializedOutputTensor(PatternRewriter &rewriter, Location loc, RankedTensorType resultType) {
    Value emptyTensor = rewriter.create<tensor::EmptyOp>(loc, resultType, ValueRange{});
    if (sparse_tensor::getSparseTensorEncoding(resultType)) {
        return emptyTensor;
    } else {
        Value zero = rewriter.create<arith::ConstantOp>(loc, rewriter.getZeroAttr(resultType.getElementType()));
        return rewriter.create<linalg::FillOp>(loc, ValueRange{zero}, ValueRange{emptyTensor}).getResult(0);
    }
}


// Helper function to map GraphBLAS Semirings to MLIR Arith operations
static Value buildSemiringOperation(OpBuilder &builder, Location loc, gblas::BinaryOp opType, Value lhs, Value rhs) {
    switch (opType) {
        case gblas::BinaryOp::plus:       return builder.create<arith::AddFOp>(loc, lhs, rhs);
        case gblas::BinaryOp::multiplies: return builder.create<arith::MulFOp>(loc, lhs, rhs);
        case gblas::BinaryOp::min:        return builder.create<arith::MinimumFOp>(loc, lhs, rhs);
        case gblas::BinaryOp::max:        return builder.create<arith::MaximumFOp>(loc, lhs, rhs);
        case gblas::BinaryOp::first:      return lhs; 
        case gblas::BinaryOp::second:     return rhs; 
        default:                          return builder.create<arith::AddFOp>(loc, lhs, rhs);
    }
}

static Value buildMathOp(OpBuilder &b, Location loc, StringRef opName, Value lhs, Value rhs = nullptr) {
    if (opName == "plus") return b.create<arith::AddFOp>(loc, lhs, rhs);
    if (opName == "minus") return b.create<arith::SubFOp>(loc, lhs, rhs);
    if (opName == "multiplies") return b.create<arith::MulFOp>(loc, lhs, rhs);
    if (opName == "div") return b.create<arith::DivFOp>(loc, lhs, rhs);
    if (opName == "second") return rhs; // Just return the right-hand value
    if (opName == "count") return b.create<arith::AddFOp>(loc, lhs, b.create<arith::ConstantOp>(loc, b.getFloatAttr(lhs.getType(), 1.0)));
    if (opName == "abs") return b.create<math::AbsFOp>(loc, lhs);
    return lhs; // Fallback
}



/**
struct FromCooOpLowering : public OpRewritePattern<gblas::FromCooOp> {
  using OpRewritePattern<gblas::FromCooOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(gblas::FromCooOp op, PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto resultType = cast<RankedTensorType>(op.getResult().getType());
    int rank = resultType.getRank();
    if (rank != 1 && rank != 2) return failure();

    bool isSparse = sparse_tensor::getSparseTensorEncoding(resultType) != nullptr;
    auto denseType = RankedTensorType::get(resultType.getShape(), resultType.getElementType());
    Value emptyTensor = rewriter.create<tensor::EmptyOp>(loc, denseType.getShape(), denseType.getElementType());
    Value zero = rewriter.create<arith::ConstantOp>(loc, rewriter.getZeroAttr(denseType.getElementType()));
    Value initTensor = rewriter.create<linalg::FillOp>(loc, ValueRange{zero}, ValueRange{emptyTensor}).getResult(0);

    Value c0 = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    Value c1 = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    Value length = rewriter.create<tensor::DimOp>(loc, op.getValues(), 0);

    auto forOp = rewriter.create<scf::ForOp>(
        loc, c0, length, c1, ValueRange{initTensor},
        [&](OpBuilder &b, Location loc, Value iv, ValueRange iterArgs) {
            Value val = b.create<tensor::ExtractOp>(loc, op.getValues(), ValueRange{iv});
            Value updatedTensor;
            if (rank == 1) {
                Value idxRaw = b.create<tensor::ExtractOp>(loc, op.getIndices(), ValueRange{iv, c0});
                Value idx = b.create<arith::IndexCastOp>(loc, b.getIndexType(), idxRaw);
                updatedTensor = b.create<tensor::InsertOp>(loc, val, iterArgs[0], ValueRange{idx});
            } else {
                Value rowRaw = b.create<tensor::ExtractOp>(loc, op.getIndices(), ValueRange{iv, c0});
                Value colRaw = b.create<tensor::ExtractOp>(loc, op.getIndices(), ValueRange{iv, c1});
                Value row = b.create<arith::IndexCastOp>(loc, b.getIndexType(), rowRaw);
                Value col = b.create<arith::IndexCastOp>(loc, b.getIndexType(), colRaw);
                updatedTensor = b.create<tensor::InsertOp>(loc, val, iterArgs[0], ValueRange{row, col});
            }
            b.create<scf::YieldOp>(loc, updatedTensor);
        });

    Value finalResult = forOp.getResult(0);
    if (isSparse) {
        finalResult = rewriter.create<sparse_tensor::ConvertOp>(loc, resultType, finalResult);
    }
    rewriter.replaceOp(op, finalResult);
    return success();
  }
};

**/

/** dynamic tensors
struct FromCooOpLowering : public OpRewritePattern<gblas::FromCooOp> {
  using OpRewritePattern<gblas::FromCooOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(gblas::FromCooOp op, PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto resultType = cast<RankedTensorType>(op.getResult().getType());
    int rank = resultType.getRank();
    if (rank != 1 && rank != 2) return failure();

    bool isSparse = sparse_tensor::getSparseTensorEncoding(resultType) != nullptr;
    auto denseType = RankedTensorType::get(resultType.getShape(), resultType.getElementType());
    
    // Extracted the dynamic sizes passed from MLIR and feed them to the EmptyOp
    ValueRange dynSizes = op.getDynamicSizes();
    Value emptyTensor = rewriter.create<tensor::EmptyOp>(loc, denseType, dynSizes);
    
    Value zero = rewriter.create<arith::ConstantOp>(loc, rewriter.getZeroAttr(denseType.getElementType()));
    Value initTensor = rewriter.create<linalg::FillOp>(loc, ValueRange{zero}, ValueRange{emptyTensor}).getResult(0);

    Value c0 = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    Value c1 = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    Value length = rewriter.create<tensor::DimOp>(loc, op.getValues(), 0);

    auto forOp = rewriter.create<scf::ForOp>(
        loc, c0, length, c1, ValueRange{initTensor},
        [&](OpBuilder &b, Location loc, Value iv, ValueRange iterArgs) {
            Value val = b.create<tensor::ExtractOp>(loc, op.getValues(), ValueRange{iv});
            Value updatedTensor;
            if (rank == 1) {
                Value idxRaw = b.create<tensor::ExtractOp>(loc, op.getIndices(), ValueRange{iv, c0});
                Value idx = b.create<arith::IndexCastOp>(loc, b.getIndexType(), idxRaw);
                updatedTensor = b.create<tensor::InsertOp>(loc, val, iterArgs[0], ValueRange{idx});
            } else {
                Value rowRaw = b.create<tensor::ExtractOp>(loc, op.getIndices(), ValueRange{iv, c0});
                Value colRaw = b.create<tensor::ExtractOp>(loc, op.getIndices(), ValueRange{iv, c1});
                Value row = b.create<arith::IndexCastOp>(loc, b.getIndexType(), rowRaw);
                Value col = b.create<arith::IndexCastOp>(loc, b.getIndexType(), colRaw);
                updatedTensor = b.create<tensor::InsertOp>(loc, val, iterArgs[0], ValueRange{row, col});
            }
            b.create<scf::YieldOp>(loc, updatedTensor);
        });

    Value finalResult = forOp.getResult(0);
    if (isSparse) {
        finalResult = rewriter.create<sparse_tensor::ConvertOp>(loc, resultType, finalResult);
    }
    rewriter.replaceOp(op, finalResult);
    return success();
  }
};

**/

struct FromCooOpLowering : public OpRewritePattern<gblas::FromCooOp> {
  using OpRewritePattern<gblas::FromCooOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(gblas::FromCooOp op, PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto resultType = cast<RankedTensorType>(op.getResult().getType());
    int rank = resultType.getRank();
    if (rank != 1 && rank != 2) return failure();

    bool isSparse = sparse_tensor::getSparseTensorEncoding(resultType) != nullptr;

    // Allocate an empty SPARSE tensor using dynamic sizes.
    // This avoids the dense OOM crash.
    Value initTensor = rewriter.create<tensor::EmptyOp>(loc, resultType, op.getDynamicSizes());

    Value c0 = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    Value c1 = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    Value length = rewriter.create<tensor::DimOp>(loc, op.getValues(), 0);

    // Loop over the COO arrays and insert directly into the sparse tensor
    auto forOp = rewriter.create<scf::ForOp>(
        loc, c0, length, c1, ValueRange{initTensor},
        [&](OpBuilder &b, Location loc, Value iv, ValueRange iterArgs) {
            Value val = b.create<tensor::ExtractOp>(loc, op.getValues(), ValueRange{iv});
            Value updatedTensor;
            
            if (rank == 1) {
                Value idxRaw = b.create<tensor::ExtractOp>(loc, op.getIndices(), ValueRange{iv, c0});
                Value idx = b.create<arith::IndexCastOp>(loc, b.getIndexType(), idxRaw);
                updatedTensor = b.create<tensor::InsertOp>(loc, val, iterArgs[0], ValueRange{idx});
            } else {
                Value rowRaw = b.create<tensor::ExtractOp>(loc, op.getIndices(), ValueRange{iv, c0});
                Value colRaw = b.create<tensor::ExtractOp>(loc, op.getIndices(), ValueRange{iv, c1});
                Value row = b.create<arith::IndexCastOp>(loc, b.getIndexType(), rowRaw);
                Value col = b.create<arith::IndexCastOp>(loc, b.getIndexType(), colRaw);
                updatedTensor = b.create<tensor::InsertOp>(loc, val, iterArgs[0], ValueRange{row, col});
            }
            b.create<scf::YieldOp>(loc, updatedTensor);
        });

    Value finalResult = forOp.getResult(0);

    // Finalize the sparse tensor.
    if (isSparse) {
        finalResult = rewriter.create<sparse_tensor::LoadOp>(
            loc, resultType, finalResult, true);
    }

    rewriter.replaceOp(op, finalResult);
    return success();
  }
};


struct TransposeOpLowering : public OpRewritePattern<gblas::TransposeOp> {
  using OpRewritePattern<gblas::TransposeOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(gblas::TransposeOp op, PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    // FIXED: Use llvm::cast instead of .cast()
    auto outputType = llvm::cast<RankedTensorType>(op.getType());

    Value outTensor = getInitializedOutputTensor(rewriter, loc, outputType);

    AffineMap inputMap = AffineMap::get(2, 0, {rewriter.getAffineDimExpr(1), rewriter.getAffineDimExpr(0)}, rewriter.getContext());
    AffineMap outputMap = rewriter.getMultiDimIdentityMap(2);

    auto genericOp = rewriter.create<linalg::GenericOp>(
        // FIXED: Use TypeRange(outputType) with parentheses
        loc, TypeRange(outputType), ValueRange{op.getInput()}, ValueRange{outTensor},
        ArrayRef<AffineMap>{inputMap, outputMap},
        ArrayRef<utils::IteratorType>{utils::IteratorType::parallel, utils::IteratorType::parallel},
        [&](OpBuilder &b, Location nestedLoc, ValueRange args) {
          b.create<linalg::YieldOp>(nestedLoc, args[0]);
        }
    );

    rewriter.replaceOp(op, genericOp.getResult(0));
    return success();
  }
};



// gblas.intersect Lowering (Using BinaryOp Enum)
struct IntersectOpLowering : public OpRewritePattern<gblas::IntersectOp> {
  using OpRewritePattern<gblas::IntersectOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(gblas::IntersectOp op, PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto outputType = cast<RankedTensorType>(op.getType());
    int64_t rank = outputType.getRank();

    bool hasMask = op.getMask() != nullptr;
    SmallVector<Value> inputs = {op.getA(), op.getB()};
    if (hasMask) inputs.push_back(op.getMask());

    // 1. Collect runtime sizes for any dynamic dimensions from Input A
    SmallVector<Value> dynamicSizes;
    for (int64_t i = 0; i < rank; ++i) {
        if (outputType.isDynamicDim(i)) {
            Value dimIdx = rewriter.create<arith::ConstantIndexOp>(loc, i);
            Value dynSize = rewriter.create<tensor::DimOp>(loc, op.getA(), dimIdx);
            dynamicSizes.push_back(dynSize);
        }
    }

    // 2. Create the empty output tensor with dynamic sizes
    Value outTensor = rewriter.create<tensor::EmptyOp>(
        loc, outputType.getShape(), outputType.getElementType(), dynamicSizes
    );

    SmallVector<AffineMap> maps(inputs.size() + 1, rewriter.getMultiDimIdentityMap(rank));
    SmallVector<utils::IteratorType> iterTypes(rank, utils::IteratorType::parallel);
 
    auto genericOp = rewriter.create<linalg::GenericOp>(
        loc, TypeRange(outputType), inputs, ValueRange{outTensor}, maps, iterTypes,
        [&](OpBuilder &b, Location nestedLoc, ValueRange args) {
            Value valA = args[0];
            Value valB = args[1];

            // 1. Fetch the Enum
            gblas::BinaryOp opType = op.getIntersectOperator();

            // 2. Force Structural Intersection using sparse_tensor.binary
            auto binOp = b.create<sparse_tensor::BinaryOp>(nestedLoc, valA.getType(), valA, valB);
            
            // Create the Overlap region (computes ONLY when BOTH A and B are present)
            Block *overlapBlock = b.createBlock(&binOp.getOverlapRegion(), {}, 
                                                {valA.getType(), valB.getType()}, 
                                                {nestedLoc, nestedLoc});
            
            // Build the actual math (minimumf, addf, etc.) inside this overlap block
            Value innerRes = buildSemiringOperation(b, nestedLoc, opType, 
                                                    overlapBlock->getArgument(0), 
                                                    overlapBlock->getArgument(1));
            b.create<sparse_tensor::YieldOp>(nestedLoc, innerRes);
            
            // Move the builder back to the main linalg.generic block
            b.setInsertionPointAfter(binOp);
            Value result = binOp.getResult();

            // 3. Apply masking if present 
            if (hasMask) {
                Value maskVal = args[2];
                Type resType = result.getType();
                
                Value maskFloat = maskVal;
                if (maskVal.getType().isIntOrIndex()) {
                    maskFloat = b.create<arith::UIToFPOp>(nestedLoc, resType, maskVal);
                }

                if (op.getMaskComplement()) {
                    Value one = b.create<arith::ConstantOp>(nestedLoc, b.getFloatAttr(resType, 1.0));
                    maskFloat = b.create<arith::SubFOp>(nestedLoc, one, maskFloat);
                }

                // mulf remains perfectly zero-preserving for the mask
                result = b.create<arith::MulFOp>(nestedLoc, result, maskFloat);
            }

            b.create<linalg::YieldOp>(nestedLoc, result);
        }
    );

    rewriter.replaceOp(op, genericOp.getResult(0));
    return success();
  }
};

/** 
struct VxmOpLowering : public OpRewritePattern<gblas::VxmOp> {

  using OpRewritePattern<gblas::VxmOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(gblas::VxmOp op, PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto resultType = cast<RankedTensorType>(op.getResult().getType());
    MLIRContext *context = rewriter.getContext();
    
    // Setup dimensions for the 2D VXM loop
    AffineExpr n, k; 
    bindDims(context, n, k);
    
    // ==========================================
    // STEP 1: PURE MATH (For the Sparsifier)
    // ==========================================
    SmallVector<Value> mathInputs = {op.getVector(), op.getMatrix()};
    SmallVector<AffineMap> mathMaps = { 
        AffineMap::get(2, 0, {k}, context),     // Vector
        AffineMap::get(2, 0, {k, n}, context),  // Matrix
        AffineMap::get(2, 0, {n}, context)      // Outs (Accumulator)
    };
    SmallVector<utils::IteratorType> mathIterators = {
        utils::IteratorType::parallel, 
        utils::IteratorType::reduction
    };

    auto mathOp = rewriter.create<linalg::GenericOp>(
        loc, resultType, mathInputs, ValueRange{op.getOuts()}, mathMaps, mathIterators,
        [&](OpBuilder &b, Location loc, ValueRange args) {
            // args[0] = vector, args[1] = matrix, args[2] = outs
            Value combined = buildSemiringOperation(b, loc, op.getCombineOp(), args[0], args[1]);
            Value reduced = buildSemiringOperation(b, loc, op.getReduceOp(), combined, args[2]);
            b.create<linalg::YieldOp>(loc, reduced);
        });

    // If no mask was provided in the IR, we are done!
    if (!op.getMask()) {
        rewriter.replaceOp(op, mathOp.getResult(0));
        return success();
    }

    // ==========================================
    // STEP 2: DENSE LOGIC (For Standard LLVM)
    // ==========================================
    // This is a 1D loop evaluating the mask element-by-element
    AffineExpr d0; 
    bindDims(context, d0);
    
    SmallVector<Value> maskInputs = {mathOp.getResult(0), op.getMask()};
    SmallVector<AffineMap> maskMaps = {
        AffineMap::get(1, 0, {d0}, context), // Input 1: Newly computed math values
        AffineMap::get(1, 0, {d0}, context), // Input 2: The Mask
        AffineMap::get(1, 0, {d0}, context)  // Outs: Original state to fall back on
    };
    SmallVector<utils::IteratorType> maskIterators = {utils::IteratorType::parallel};

    auto maskOp = rewriter.create<linalg::GenericOp>(
        loc, resultType, maskInputs, ValueRange{op.getOuts()}, maskMaps, maskIterators,
        [&](OpBuilder &b, Location loc, ValueRange args) {
            Value computed_val = args[0];
            Value mask_val = args[1];
            Value original_val = args[2]; // op.getOuts()

            // 1. Check if mask value > 0
            Value mask_cond = b.create<arith::CmpFOp>(
                loc, arith::CmpFPredicate::UGT, mask_val, 
                b.create<arith::ConstantOp>(loc, b.getFloatAttr(b.getF32Type(), 0.0))
            );
            
            // 2. Invert logic if complement is true
            if (op.getMaskComplement()) {
                mask_cond = b.create<arith::XOrIOp>(
                    loc, mask_cond, 
                    b.create<arith::ConstantIntOp>(loc, 1, 1) // true
                );
            }
            
            // 3. Select computed value if condition is met, else revert to original
            Value selected = b.create<arith::SelectOp>(loc, mask_cond, computed_val, original_val);
            b.create<linalg::YieldOp>(loc, selected);
        });

    // Replace the original gblas.vxm with the final masked output
    rewriter.replaceOp(op, maskOp.getResult(0));
    return success();
  }
};

**/


struct VxmOpLowering : public OpRewritePattern<gblas::VxmOp> {
  using OpRewritePattern<gblas::VxmOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(gblas::VxmOp op, PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto resultType = cast<RankedTensorType>(op.getResult().getType());
    MLIRContext *context = rewriter.getContext();
    
    // Setup dimensions for the 2D VXM loop
    AffineExpr n, k; 
    bindDims(context, n, k);
    
    // PURE MATH (For the Sparsifier)
    SmallVector<Value> mathInputs = {op.getVector(), op.getMatrix()};
    SmallVector<AffineMap> mathMaps = { 
        AffineMap::get(2, 0, {k}, context), // Vector
        AffineMap::get(2, 0, {k, n}, context), // Matrix
        AffineMap::get(2, 0, {n}, context) // Outs (Accumulator)
    };
    SmallVector<utils::IteratorType> mathIterators = {
        utils::IteratorType::parallel, 
        utils::IteratorType::reduction
    };

    auto mathOp = rewriter.create<linalg::GenericOp>(
        loc, resultType, mathInputs, ValueRange{op.getOuts()}, mathMaps, mathIterators,
        [&](OpBuilder &b, Location loc, ValueRange args) {
            // args[0] = vector, args[1] = matrix, args[2] = outs
            Value combined = buildSemiringOperation(b, loc, op.getCombineOp(), args[0], args[1]);
            Value reduced = buildSemiringOperation(b, loc, op.getReduceOp(), combined, args[2]);
            b.create<linalg::YieldOp>(loc, reduced);
        });

    // If no mask was provided in the IR, we are done!
    if (!op.getMask()) {
        rewriter.replaceOp(op, mathOp.getResult(0));
        return success();
    }

    // DENSE LOGIC (For Standard LLVM)
    // This is a 1D loop evaluating the mask element-by-element
    AffineExpr d0; 
    bindDims(context, d0);
    
    SmallVector<Value> maskInputs = {mathOp.getResult(0), op.getMask()};
    SmallVector<AffineMap> maskMaps = {
        AffineMap::get(1, 0, {d0}, context), // Input 1: Newly computed math values
        AffineMap::get(1, 0, {d0}, context), // Input 2: The Mask
        AffineMap::get(1, 0, {d0}, context) // Outs: Original state to fall back on
    };
    SmallVector<utils::IteratorType> maskIterators = {utils::IteratorType::parallel};

    auto maskOp = rewriter.create<linalg::GenericOp>(
        loc, resultType, maskInputs, ValueRange{op.getOuts()}, maskMaps, maskIterators,
        [&](OpBuilder &b, Location loc, ValueRange args) {
            Value computed_val = args[0];
            Value mask_val = args[1];
            Value original_val = args[2]; // op.getOuts()

            // 1. Check if mask value > 0
            Value mask_cond = b.create<arith::CmpFOp>(
                loc, arith::CmpFPredicate::UGT, mask_val, 
                b.create<arith::ConstantOp>(loc, b.getFloatAttr(b.getF32Type(), 0.0))
            );
            
            // 2. Invert logic if complement is true
            if (op.getMaskComplement()) {
                mask_cond = b.create<arith::XOrIOp>(
                    loc, mask_cond, 
                    b.create<arith::ConstantIntOp>(loc, 1, 1) // true
                );
            }
            
            // 3. Select computed value if condition is met, else revert to original
            Value selected = b.create<arith::SelectOp>(loc, mask_cond, computed_val, original_val);
            b.create<linalg::YieldOp>(loc, selected);
        });

    // Replace the original gblas.vxm with the final masked output
    rewriter.replaceOp(op, maskOp.getResult(0));
    return success();
  }
};




struct MxvOpLowering : public OpRewritePattern<gblas::MxvOp> {
  using OpRewritePattern<gblas::MxvOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(gblas::MxvOp op, PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto resultType = cast<RankedTensorType>(op.getResult().getType());
    Value outTensor = getInitializedOutputTensor(rewriter, loc, resultType);
    MLIRContext *context = rewriter.getContext();
    AffineExpr m, k; bindDims(context, m, k);
    SmallVector<Value> inputs = {op.getMatrix(), op.getVector()};
    SmallVector<AffineMap> indexingMaps = {AffineMap::get(2, 0, {m, k}, context), AffineMap::get(2, 0, {k}, context)};
    if (op.getMask()) {
        inputs.push_back(op.getMask());
        indexingMaps.push_back(AffineMap::get(2, 0, {m}, context));
    }
    indexingMaps.push_back(AffineMap::get(2, 0, {m}, context));
    SmallVector<utils::IteratorType> iteratorTypes = {utils::IteratorType::parallel, utils::IteratorType::reduction};
    auto genericOp = rewriter.create<linalg::GenericOp>(
        loc, resultType, inputs, ValueRange{outTensor}, indexingMaps, iteratorTypes,
        [&](OpBuilder &b, Location loc, ValueRange args) {
            Value combined = buildSemiringOperation(b, loc, op.getCombineOp(), args[0], args[1]);
            Value out_val = op.getMask() ? args[3] : args[2];
            Value reduced = buildSemiringOperation(b, loc, op.getReduceOp(), combined, out_val);
            if (op.getMask()) {
                Value mask_cond = b.create<arith::CmpFOp>(loc, arith::CmpFPredicate::UGT, args[2], b.create<arith::ConstantOp>(loc, b.getFloatAttr(b.getF32Type(), 0.0)));
                if (op.getMaskComplement()) mask_cond = b.create<arith::XOrIOp>(loc, mask_cond, b.create<arith::ConstantIntOp>(loc, 1, 1));
                b.create<linalg::YieldOp>(loc, b.create<arith::SelectOp>(loc, mask_cond, reduced, out_val).getResult());
            } else { b.create<linalg::YieldOp>(loc, reduced); }
        });
    rewriter.replaceOp(op, genericOp.getResult(0));
    return success();
  }
};


struct MxmOpLowering : public OpRewritePattern<gblas::MxmOp> {
  using OpRewritePattern<gblas::MxmOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(gblas::MxmOp op, PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto resultType = cast<RankedTensorType>(op.getResult().getType());
    
    // =====================================================================
    // THE FIX: Dynamically extract sizes for tensor.empty
    // =====================================================================
    SmallVector<Value> dynSizes;
    
    // 1. If output dimension 0 is dynamic, it matches the rows (dim 0) of A
    if (resultType.isDynamicDim(0)) {
        dynSizes.push_back(rewriter.create<tensor::DimOp>(loc, op.getA(), 0));
    }
    
    // 2. If output dimension 1 is dynamic, it matches the cols (dim 1) of B
    if (resultType.isDynamicDim(1)) {
        dynSizes.push_back(rewriter.create<tensor::DimOp>(loc, op.getB(), 1));
    }

    // 3. Create the empty tensor using the dynamically extracted sizes.
    Value outTensor = rewriter.create<tensor::EmptyOp>(loc, resultType, dynSizes);

    // --- FIX: Zero out uninitialized memory for dense tensors ---
    if (!sparse_tensor::getSparseTensorEncoding(resultType)) {
        Value zero = rewriter.create<arith::ConstantOp>(loc, rewriter.getZeroAttr(resultType.getElementType()));
        outTensor = rewriter.create<linalg::FillOp>(loc, ValueRange{zero}, ValueRange{outTensor}).getResult(0);
    }
    // ------------------------------------------------------------


    MLIRContext *context = rewriter.getContext();
    AffineExpr m, n, k; 
    bindDims(context, m, n, k);
    
    SmallVector<Value> inputs = {op.getA(), op.getB()};
    SmallVector<AffineMap> indexingMaps = {
        AffineMap::get(3, 0, {m, k}, context), 
        AffineMap::get(3, 0, {k, n}, context)
    };
    
    if (op.getMask()) {
        inputs.push_back(op.getMask());
        indexingMaps.push_back(AffineMap::get(3, 0, {m, n}, context));
    }
    
    // The indexing map for the output tensor
    indexingMaps.push_back(AffineMap::get(3, 0, {m, n}, context));
    
    SmallVector<utils::IteratorType> iteratorTypes = {
        utils::IteratorType::parallel, 
        utils::IteratorType::parallel, 
        utils::IteratorType::reduction
    };
    
    // auto genericOp = rewriter.create<linalg::GenericOp>(
    //     loc, resultType, inputs, ValueRange{outTensor}, indexingMaps, iteratorTypes,
    //     [&](OpBuilder &b, Location loc, ValueRange args) {
            
    //         // 1. Compute A * B
    //         Value combined = buildSemiringOperation(b, loc, op.getCombineOp(), args[0], args[1]);
    //         Value out_val = op.getMask() ? args[3] : args[2];
            
    //         // 2. Algebraic Masking 
    //         if (op.getMask()) {
    //             Value mask_val = args[2];
    //             // If complement is requested, invert the mask: (1.0 - mask)
    //             if (op.getMaskComplement()) {
    //                 Value one = b.create<arith::ConstantOp>(loc, b.getFloatAttr(b.getF32Type(), 1.0));
    //                 mask_val = b.create<arith::SubFOp>(loc, one, mask_val);
    //             }
    //             // Multiply the combined result by the mask
    //             combined = b.create<arith::MulFOp>(loc, combined, mask_val);
    //         }
            
    //         // 3. Accumulate: out = out + combined
    //         Value reduced = buildSemiringOperation(b, loc, op.getReduceOp(), combined, out_val);
    //         b.create<linalg::YieldOp>(loc, reduced);
    //     });
        

    auto genericOp = rewriter.create<linalg::GenericOp>(
        loc, resultType, inputs, ValueRange{outTensor}, indexingMaps, iteratorTypes,
        [&](OpBuilder &b, Location loc, ValueRange args) {
            
            Value valA = args[0];
            Value valB = args[1];
            Value out_val = op.getMask() ? args[3] : args[2];
            
            // 1. Structural Masking Injection
            if (op.getMask()) {
                Value mask_val = args[2];
                
                if (op.getMaskComplement()) {
                    // Fallback for complement: Must evaluate fully, then invert and mask
                    Value one = b.create<arith::ConstantOp>(loc, b.getFloatAttr(b.getF32Type(), 1.0));
                    mask_val = b.create<arith::SubFOp>(loc, one, mask_val);
                    Value combined = buildSemiringOperation(b, loc, op.getCombineOp(), valA, valB);
                    combined = b.create<arith::MulFOp>(loc, combined, mask_val);
                    Value reduced = buildSemiringOperation(b, loc, op.getReduceOp(), combined, out_val);
                    b.create<linalg::YieldOp>(loc, reduced);
                    return;
                } else {
                    // THE FIX: Pre-multiply A by the mask BEFORE the combine op.
                    // This forces the MLIR sparsifier to intersect the sparse coordinates 
                    // of A and the Mask on the m & n loops, preventing dense intermediate fill-in.
                    valA = b.create<arith::MulFOp>(loc, valA, mask_val);
                }
            }
            
            // 2. Compute structurally restricted A * B
            Value combined = buildSemiringOperation(b, loc, op.getCombineOp(), valA, valB);
            
            // 3. Accumulate: out = out + combined
            Value reduced = buildSemiringOperation(b, loc, op.getReduceOp(), combined, out_val);
            b.create<linalg::YieldOp>(loc, reduced);
        });


    rewriter.replaceOp(op, genericOp.getResult(0));
    return success();
  }
};



/** 
struct MxmOpLowering : public OpRewritePattern<gblas::MxmOp> {
  using OpRewritePattern<gblas::MxmOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(gblas::MxmOp op, PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto resultType = llvm::cast<mlir::ShapedType>(op.getResult().getType());

    MLIRContext *context = rewriter.getContext();
    AffineExpr m, n, k; 
    bindDims(context, m, n, k);
    
    // 1. Output Tensor Allocation
    SmallVector<Value> dynSizes;
    if (resultType.isDynamicDim(0)) dynSizes.push_back(rewriter.create<tensor::DimOp>(loc, op.getA(), 0));
    if (resultType.isDynamicDim(1)) dynSizes.push_back(rewriter.create<tensor::DimOp>(loc, op.getB(), 1));
    
    Value outTensor = rewriter.create<tensor::EmptyOp>(loc, resultType, dynSizes);
    
    bool isSparseOut = sparse_tensor::getSparseTensorEncoding(resultType) != nullptr;
    bool hasMask = op.getMask() != nullptr;
    bool isComplement = op.getMaskComplement();
    
    // --- 2. Structural Masking & Initialization ---
    if (hasMask && !isComplement && isSparseOut) {
        // STRUCTURAL MASKING: Prevent OOM by initializing the sparse output 
        // with the exact sparsity pattern of the mask, filled with zeros.
        // This forces the Sparsifier to use an in-place update and completely 
        // bypass the full A x B iteration.
        Value zero = rewriter.create<arith::ConstantOp>(loc, rewriter.getZeroAttr(resultType.getElementType()));
        SmallVector<AffineMap> map = {
            AffineMap::get(2, 0, {m, n}, context), // ins: mask
            AffineMap::get(2, 0, {m, n}, context)  // outs: empty outTensor
        };
        SmallVector<utils::IteratorType> iters = {
            utils::IteratorType::parallel, utils::IteratorType::parallel
        };
        outTensor = rewriter.create<linalg::GenericOp>(
            loc, resultType, ValueRange{op.getMask()}, ValueRange{outTensor}, map, iters,
            [&](OpBuilder &b, Location loc, ValueRange args) {
                b.create<linalg::YieldOp>(loc, zero);
            }).getResult(0);
    } else if (!isSparseOut) {
        // Dense output needs explicit background fill
        Value zero = rewriter.create<arith::ConstantOp>(loc, rewriter.getZeroAttr(resultType.getElementType()));
        outTensor = rewriter.create<linalg::FillOp>(loc, ValueRange{zero}, ValueRange{outTensor}).getResult(0);
    }

    // --- 3. Main MatMul Generic Op ---
    SmallVector<Value> inputs = {op.getA(), op.getB()};
    SmallVector<AffineMap> indexingMaps = {
        AffineMap::get(3, 0, {m, k}, context), 
        AffineMap::get(3, 0, {k, n}, context)
    };
    
    // Only pass mask as an input for ALGEBRAIC masking (complement or dense out)
    bool useAlgebraicMask = hasMask && (isComplement || !isSparseOut);
    
    if (useAlgebraicMask) {
        inputs.push_back(op.getMask());
        indexingMaps.push_back(AffineMap::get(3, 0, {m, n}, context));
    }
    
    indexingMaps.push_back(AffineMap::get(3, 0, {m, n}, context)); // outs
    
    SmallVector<utils::IteratorType> iteratorTypes = {
        utils::IteratorType::parallel, 
        utils::IteratorType::parallel, 
        utils::IteratorType::reduction
    };
    
    auto genericOp = rewriter.create<linalg::GenericOp>(
        loc, resultType, inputs, ValueRange{outTensor}, indexingMaps, iteratorTypes,
        [&](OpBuilder &b, Location loc, ValueRange args) {
            
            Value combined = buildSemiringOperation(b, loc, op.getCombineOp(), args[0], args[1]);
            
            // Apply algebraic masking only if structural masking couldn't be used
            if (useAlgebraicMask) {
                Value mask_val = args[2];
                
                if (mask_val.getType() != combined.getType()) {
                    if (llvm::isa<FloatType>(combined.getType())) {
                        if (mask_val.getType().isInteger(1)) {
                            mask_val = b.create<arith::UIToFPOp>(loc, combined.getType(), mask_val);
                        } else {
                            mask_val = b.create<arith::SIToFPOp>(loc, combined.getType(), mask_val);
                        }
                    } else if (llvm::isa<IntegerType>(combined.getType())) {
                        mask_val = b.create<arith::ExtUIOp>(loc, combined.getType(), mask_val);
                    }
                }

                if (isComplement) {
                    if (llvm::isa<FloatType>(combined.getType())) {
                        Value one = b.create<arith::ConstantOp>(loc, b.getFloatAttr(combined.getType(), 1.0));
                        mask_val = b.create<arith::SubFOp>(loc, one, mask_val);
                    } else {
                        Value one = b.create<arith::ConstantOp>(loc, b.getIntegerAttr(combined.getType(), 1));
                        mask_val = b.create<arith::SubIOp>(loc, one, mask_val);
                    }
                }
                
                if (llvm::isa<FloatType>(combined.getType())) {
                    combined = b.create<arith::MulFOp>(loc, combined, mask_val);
                } else {
                    combined = b.create<arith::MulIOp>(loc, combined, mask_val);
                }
            }
            
            Value out_val = useAlgebraicMask ? args[3] : args[2];
            Value reduced = buildSemiringOperation(b, loc, op.getReduceOp(), combined, out_val);
            b.create<linalg::YieldOp>(loc, reduced);
        });
        
    rewriter.replaceOp(op, genericOp.getResult(0));
    return success();
  }
};

**/


struct VxvOpLowering : public OpRewritePattern<gblas::VxvOp> {
  using OpRewritePattern<gblas::VxvOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(gblas::VxvOp op, PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto resultType = cast<RankedTensorType>(op.getResult().getType());
    Value outTensor = getInitializedOutputTensor(rewriter, loc, resultType);
    MLIRContext *context = rewriter.getContext();
    AffineExpr k; bindDims(context, k);
    SmallVector<Value> inputs = {op.getA(), op.getB()};
    SmallVector<AffineMap> indexingMaps = {
        AffineMap::get(1, 0, {k}, context), 
        AffineMap::get(1, 0, {k}, context),
        AffineMap::get(1, 0, {}, context)
    };
    SmallVector<utils::IteratorType> iteratorTypes = {utils::IteratorType::reduction};
    auto genericOp = rewriter.create<linalg::GenericOp>(
        loc, resultType, inputs, ValueRange{outTensor}, indexingMaps, iteratorTypes,
        [&](OpBuilder &b, Location loc, ValueRange args) {
            Value combined = buildSemiringOperation(b, loc, op.getCombineOp(), args[0], args[1]);
            Value reduced = buildSemiringOperation(b, loc, op.getReduceOp(), combined, args[2]);
            b.create<linalg::YieldOp>(loc, reduced);
        });
    rewriter.replaceOp(op, genericOp.getResult(0));
    return success();
  }
};



struct EWiseAddLowering : public OpConversionPattern<gblas::EWiseAddOp> {
  using OpConversionPattern<gblas::EWiseAddOp>::OpConversionPattern;
  LogicalResult matchAndRewrite(gblas::EWiseAddOp op, OpAdaptor adaptor, ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto resultType = llvm::cast<RankedTensorType>(op.getResult().getType());
    auto allocOp = rewriter.create<bufferization::AllocTensorOp>(loc, resultType, ValueRange{}, Value(), IntegerAttr());
    rewriter.replaceOpWithNewOp<linalg::AddOp>(op, resultType, ValueRange{adaptor.getLhs(), adaptor.getRhs()}, ValueRange{allocOp.getResult()});
    return success();
  }
};

struct NRowsLowering : public OpConversionPattern<gblas::NRowsOp> {
  using OpConversionPattern<gblas::NRowsOp>::OpConversionPattern;
  LogicalResult matchAndRewrite(gblas::NRowsOp op, OpAdaptor adaptor, ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    Value input = adaptor.getInput();
    Type type = input.getType();
    if (auto rankedType = llvm::dyn_cast<RankedTensorType>(type)) {
      Value zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
      rewriter.replaceOpWithNewOp<tensor::DimOp>(op, input, zero);
      return success();
    }
    if (auto unrankedType = llvm::dyn_cast<UnrankedTensorType>(type)) {
      Value zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
      Value rank = rewriter.create<tensor::RankOp>(loc, input);
      Value isValid = rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ugt, rank, zero);
      Value dimSize = rewriter.create<tensor::DimOp>(loc, input, zero);
      rewriter.replaceOpWithNewOp<arith::SelectOp>(op, isValid, dimSize, zero);
      return success();
    }
    return failure();
  }
};

struct NColsLowering : public OpConversionPattern<gblas::NColsOp> {
  using OpConversionPattern<gblas::NColsOp>::OpConversionPattern;
  LogicalResult matchAndRewrite(gblas::NColsOp op, OpAdaptor adaptor, ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    Value input = adaptor.getInput();
    Type type = input.getType();
    if (auto rankedType = llvm::dyn_cast<RankedTensorType>(type)) {
      if (rankedType.getRank() < 2) return failure();
      Value one = rewriter.create<arith::ConstantIndexOp>(loc, 1);
      rewriter.replaceOpWithNewOp<tensor::DimOp>(op, input, one);
      return success();
    }
    if (auto unrankedType = llvm::dyn_cast<UnrankedTensorType>(type)) {
      Value zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
      Value one = rewriter.create<arith::ConstantIndexOp>(loc, 1);
      Value rank = rewriter.create<tensor::RankOp>(loc, input);
      Value isValid = rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ugt, rank, one);
      Value dimSize = rewriter.create<tensor::DimOp>(loc, input, one);
      rewriter.replaceOpWithNewOp<arith::SelectOp>(op, isValid, dimSize, zero);
      return success();
    }
    return failure();
  }
};


/** 
struct UpdateOpLowering : public OpConversionPattern<gblas::UpdateOp> {
  using OpConversionPattern<gblas::UpdateOp>::OpConversionPattern;
  LogicalResult matchAndRewrite(gblas::UpdateOp op, OpAdaptor adaptor, ConversionPatternRewriter &rewriter) const override {
    auto output = adaptor.getOutput(); 
    auto input = adaptor.getInput();
    Type resType = output.getType(); 
    std::string accOp = op.getAccumulateOperator() ? op.getAccumulateOperator()->str() : "plus";

    if (op.getReplace()) {
        if (input.getType() == resType) {
            rewriter.replaceOp(op, input);
        } else {
            rewriter.replaceOpWithNewOp<sparse_tensor::ConvertOp>(op, resType, input);
        }
        return success();
    }

    Location loc = op.getLoc();
    if (accOp == "plus") {
        rewriter.replaceOpWithNewOp<linalg::AddOp>(op, resType, ValueRange{input, output}, ValueRange{output});
    } else if (accOp == "max") {
        auto inputType = llvm::cast<RankedTensorType>(op.getOperand(0).getType());
        int64_t rank = inputType.getRank();
        
        SmallVector<AffineMap> indexingMaps(3, rewriter.getMultiDimIdentityMap(rank));
        SmallVector<utils::IteratorType> iteratorTypes(rank, utils::IteratorType::parallel);
        SmallVector<Value, 2> inputValues = {op.getOperand(0), op.getOperand(1)};
        SmallVector<Value, 1> outputValues = {op.getOperand(1)};
        Type resultType = outputValues[0].getType();

        auto genericOp = rewriter.create<linalg::GenericOp>(
            loc, TypeRange(resultType), ValueRange(inputValues), ValueRange(outputValues),
            indexingMaps, iteratorTypes,
            [&](OpBuilder &b, Location loc, ValueRange args) {
                Value inVal = args[0]; 
                Value outVal = args[1];
                auto binaryOp = b.create<sparse_tensor::BinaryOp>(loc, inVal.getType(), inVal, outVal);

                Block *overlapBlock = b.createBlock(&binaryOp.getOverlapRegion());
                overlapBlock->addArgument(inVal.getType(), loc);
                overlapBlock->addArgument(outVal.getType(), loc);
                Value maxVal = b.create<arith::MaximumFOp>(loc, overlapBlock->getArgument(0), overlapBlock->getArgument(1));
                b.create<sparse_tensor::YieldOp>(loc, maxVal);

                Block *leftBlock = b.createBlock(&binaryOp.getLeftRegion());
                leftBlock->addArgument(inVal.getType(), loc);
                b.create<sparse_tensor::YieldOp>(loc, leftBlock->getArgument(0));

                Block *rightBlock = b.createBlock(&binaryOp.getRightRegion());
                rightBlock->addArgument(outVal.getType(), loc);
                b.create<sparse_tensor::YieldOp>(loc, rightBlock->getArgument(0));

                b.setInsertionPointAfter(binaryOp);
                b.create<linalg::YieldOp>(loc, binaryOp.getResult());
            });

        rewriter.replaceOp(op, genericOp.getResults());
    } else if (accOp == "min") {
        auto inputType = llvm::cast<RankedTensorType>(op.getOperand(0).getType());
        int64_t rank = inputType.getRank();
        
        SmallVector<AffineMap> indexingMaps(3, rewriter.getMultiDimIdentityMap(rank));
        SmallVector<utils::IteratorType> iteratorTypes(rank, utils::IteratorType::parallel);
        SmallVector<Value, 2> inputValues = {op.getOperand(0), op.getOperand(1)};
        SmallVector<Value, 1> outputValues = {op.getOperand(1)};
        Type resultType = outputValues[0].getType();

        auto genericOp = rewriter.create<linalg::GenericOp>(
            loc, TypeRange(resultType), ValueRange(inputValues), ValueRange(outputValues),
            indexingMaps, iteratorTypes,
            [&](OpBuilder &b, Location loc, ValueRange args) {
                Value inVal = args[0]; 
                Value outVal = args[1];
                auto binaryOp = b.create<sparse_tensor::BinaryOp>(loc, inVal.getType(), inVal, outVal);

                Block *overlapBlock = b.createBlock(&binaryOp.getOverlapRegion());
                overlapBlock->addArgument(inVal.getType(), loc);
                overlapBlock->addArgument(outVal.getType(), loc);
                
                // --- THE ONLY CHANGE: Use MinimumFOp ---
                Value minVal = b.create<arith::MinimumFOp>(loc, overlapBlock->getArgument(0), overlapBlock->getArgument(1));
                b.create<sparse_tensor::YieldOp>(loc, minVal);

                Block *leftBlock = b.createBlock(&binaryOp.getLeftRegion());
                leftBlock->addArgument(inVal.getType(), loc);
                b.create<sparse_tensor::YieldOp>(loc, leftBlock->getArgument(0));

                Block *rightBlock = b.createBlock(&binaryOp.getRightRegion());
                rightBlock->addArgument(outVal.getType(), loc);
                b.create<sparse_tensor::YieldOp>(loc, rightBlock->getArgument(0));

                b.setInsertionPointAfter(binaryOp);
                b.create<linalg::YieldOp>(loc, binaryOp.getResult());
            });

        rewriter.replaceOp(op, genericOp.getResults());
    } else {
        return op.emitError("unsupported accumulate operator: ") << accOp;
    }
    return success();
  }
};
**/

/**
 
struct UpdateOpLowering : public OpConversionPattern<gblas::UpdateOp> {
  using OpConversionPattern<gblas::UpdateOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(gblas::UpdateOp op, OpAdaptor adaptor, ConversionPatternRewriter &rewriter) const override {
    auto output = adaptor.getOutput(); 
    auto input = adaptor.getInput();
    Type resType = output.getType(); 
    std::string accOp = op.getAccumulateOperator() ? op.getAccumulateOperator()->str() : "plus";
    Location loc = op.getLoc();

    // Fix: If replacing, we must explicitly COPY the data into the 'output' buffer 
    // to maintain MLIR's strict Destination-Passing Style rules.
    if (op.getReplace()) {
        auto rankedType = llvm::cast<RankedTensorType>(resType);
        int64_t rank = rankedType.getRank();
        
        Value convertedInput = input;
        if (input.getType() != resType) {
            convertedInput = rewriter.create<sparse_tensor::ConvertOp>(loc, resType, input);
        }

        auto genericOp = rewriter.create<linalg::GenericOp>(
            loc, TypeRange{resType}, ValueRange{convertedInput}, ValueRange{output},
            SmallVector<AffineMap>(2, rewriter.getMultiDimIdentityMap(rank)),
            SmallVector<utils::IteratorType>(rank, utils::IteratorType::parallel),
            [&](OpBuilder &b, Location loc, ValueRange args) {
                b.create<linalg::YieldOp>(loc, args[0]);
            });
            
        rewriter.replaceOp(op, genericOp.getResult(0));
        return success();
    }

    if (accOp == "plus") {
        rewriter.replaceOpWithNewOp<linalg::AddOp>(op, resType, ValueRange{input, output}, ValueRange{output});
    } else if (accOp == "max") {
        auto inputType = llvm::cast<RankedTensorType>(op.getOperand(0).getType());
        int64_t rank = inputType.getRank();
        
        SmallVector<AffineMap> indexingMaps(3, rewriter.getMultiDimIdentityMap(rank));
        SmallVector<utils::IteratorType> iteratorTypes(rank, utils::IteratorType::parallel);
        SmallVector<Value, 2> inputValues = {op.getOperand(0), op.getOperand(1)};
        SmallVector<Value, 1> outputValues = {op.getOperand(1)};
        Type resultType = outputValues[0].getType();

        auto genericOp = rewriter.create<linalg::GenericOp>(
            loc, TypeRange(resultType), ValueRange(inputValues), ValueRange(outputValues),
            indexingMaps, iteratorTypes,
            [&](OpBuilder &b, Location loc, ValueRange args) {
                Value inVal = args[0]; 
                Value outVal = args[1];
                auto binaryOp = b.create<sparse_tensor::BinaryOp>(loc, inVal.getType(), inVal, outVal);

                Block *overlapBlock = b.createBlock(&binaryOp.getOverlapRegion());
                overlapBlock->addArgument(inVal.getType(), loc);
                overlapBlock->addArgument(outVal.getType(), loc);
                Value maxVal = b.create<arith::MaximumFOp>(loc, overlapBlock->getArgument(0), overlapBlock->getArgument(1));
                b.create<sparse_tensor::YieldOp>(loc, maxVal);

                Block *leftBlock = b.createBlock(&binaryOp.getLeftRegion());
                leftBlock->addArgument(inVal.getType(), loc);
                b.create<sparse_tensor::YieldOp>(loc, leftBlock->getArgument(0));

                Block *rightBlock = b.createBlock(&binaryOp.getRightRegion());
                rightBlock->addArgument(outVal.getType(), loc);
                b.create<sparse_tensor::YieldOp>(loc, rightBlock->getArgument(0));

                b.setInsertionPointAfter(binaryOp);
                b.create<linalg::YieldOp>(loc, binaryOp.getResult());
            });

        rewriter.replaceOp(op, genericOp.getResults());
    } else if (accOp == "min") {
        rewriter.replaceOpWithNewOp<linalg::MinOp>(op, resType, ValueRange{input, output}, ValueRange{output});
    } else {
        return op.emitError("unsupported accumulate operator: ") << accOp;
    }
    return success();
  }
};
**/


struct UpdateOpLowering : public OpConversionPattern<gblas::UpdateOp> {
  using OpConversionPattern<gblas::UpdateOp>::OpConversionPattern;
  LogicalResult matchAndRewrite(gblas::UpdateOp op, OpAdaptor adaptor, ConversionPatternRewriter &rewriter) const override {
    auto output = adaptor.getOutput(); 
    auto input = adaptor.getInput();
    Type resType = output.getType(); 
    std::string accOp = op.getAccumulateOperator() ? op.getAccumulateOperator()->str() : "plus";

    if (op.getReplace()) {
        if (input.getType() == resType) {
            rewriter.replaceOp(op, input);
        } else {
            rewriter.replaceOpWithNewOp<sparse_tensor::ConvertOp>(op, resType, input);
        }
        return success();
    }

    Location loc = op.getLoc();
    if (accOp == "plus") {
        rewriter.replaceOpWithNewOp<linalg::AddOp>(op, resType, ValueRange{input, output}, ValueRange{output});
    } else if (accOp == "max") {
        auto inputType = llvm::cast<RankedTensorType>(op.getOperand(0).getType());
        int64_t rank = inputType.getRank();
        
        SmallVector<AffineMap> indexingMaps(3, rewriter.getMultiDimIdentityMap(rank));
        SmallVector<utils::IteratorType> iteratorTypes(rank, utils::IteratorType::parallel);
        SmallVector<Value, 2> inputValues = {op.getOperand(0), op.getOperand(1)};
        SmallVector<Value, 1> outputValues = {op.getOperand(1)};
        Type resultType = outputValues[0].getType();

        auto genericOp = rewriter.create<linalg::GenericOp>(
            loc, TypeRange(resultType), ValueRange(inputValues), ValueRange(outputValues),
            indexingMaps, iteratorTypes,
            [&](OpBuilder &b, Location loc, ValueRange args) {
                Value inVal = args[0]; 
                Value outVal = args[1];
                auto binaryOp = b.create<sparse_tensor::BinaryOp>(loc, inVal.getType(), inVal, outVal);

                Block *overlapBlock = b.createBlock(&binaryOp.getOverlapRegion());
                overlapBlock->addArgument(inVal.getType(), loc);
                overlapBlock->addArgument(outVal.getType(), loc);
                Value maxVal = b.create<arith::MaximumFOp>(loc, overlapBlock->getArgument(0), overlapBlock->getArgument(1));
                b.create<sparse_tensor::YieldOp>(loc, maxVal);

                Block *leftBlock = b.createBlock(&binaryOp.getLeftRegion());
                leftBlock->addArgument(inVal.getType(), loc);
                b.create<sparse_tensor::YieldOp>(loc, leftBlock->getArgument(0));

                Block *rightBlock = b.createBlock(&binaryOp.getRightRegion());
                rightBlock->addArgument(outVal.getType(), loc);
                b.create<sparse_tensor::YieldOp>(loc, rightBlock->getArgument(0));

                b.setInsertionPointAfter(binaryOp);
                b.create<linalg::YieldOp>(loc, binaryOp.getResult());
            });

        rewriter.replaceOp(op, genericOp.getResults());
    } else if (accOp == "min") {
        auto inputType = llvm::cast<RankedTensorType>(op.getOperand(0).getType());
        int64_t rank = inputType.getRank();
        
        SmallVector<AffineMap> indexingMaps(3, rewriter.getMultiDimIdentityMap(rank));
        SmallVector<utils::IteratorType> iteratorTypes(rank, utils::IteratorType::parallel);
        SmallVector<Value, 2> inputValues = {op.getOperand(0), op.getOperand(1)};
        SmallVector<Value, 1> outputValues = {op.getOperand(1)};
        Type resultType = outputValues[0].getType();

        auto genericOp = rewriter.create<linalg::GenericOp>(
            loc, TypeRange(resultType), ValueRange(inputValues), ValueRange(outputValues),
            indexingMaps, iteratorTypes,
            [&](OpBuilder &b, Location loc, ValueRange args) {
                Value inVal = args[0]; 
                Value outVal = args[1];
                auto binaryOp = b.create<sparse_tensor::BinaryOp>(loc, inVal.getType(), inVal, outVal);

                Block *overlapBlock = b.createBlock(&binaryOp.getOverlapRegion());
                overlapBlock->addArgument(inVal.getType(), loc);
                overlapBlock->addArgument(outVal.getType(), loc);
                
                // --- THE ONLY CHANGE: Use MinimumFOp ---
                Value minVal = b.create<arith::MinimumFOp>(loc, overlapBlock->getArgument(0), overlapBlock->getArgument(1));
                b.create<sparse_tensor::YieldOp>(loc, minVal);

                Block *leftBlock = b.createBlock(&binaryOp.getLeftRegion());
                leftBlock->addArgument(inVal.getType(), loc);
                b.create<sparse_tensor::YieldOp>(loc, leftBlock->getArgument(0));

                Block *rightBlock = b.createBlock(&binaryOp.getRightRegion());
                rightBlock->addArgument(outVal.getType(), loc);
                b.create<sparse_tensor::YieldOp>(loc, rightBlock->getArgument(0));

                b.setInsertionPointAfter(binaryOp);
                b.create<linalg::YieldOp>(loc, binaryOp.getResult());
            });

        rewriter.replaceOp(op, genericOp.getResults());
    } else {
        return op.emitError("unsupported accumulate operator: ") << accOp;
    }
    return success();
  }
};




#include "mlir/IR/BuiltinOps.h" // Required for UnrealizedConversionCastOp



struct ToPtrOpLowering : public mlir::OpRewritePattern<gblas::ToPtrOp> {
  using OpRewritePattern<gblas::ToPtrOp>::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(gblas::ToPtrOp op, 
                                      mlir::PatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<mlir::UnrealizedConversionCastOp>(
        op, op.getOutput().getType(), op.getInput());
    return mlir::success();
  }
};

struct FromPtrOpLowering : public mlir::OpRewritePattern<gblas::FromPtrOp> {
  using OpRewritePattern<gblas::FromPtrOp>::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(gblas::FromPtrOp op, 
                                      mlir::PatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<mlir::UnrealizedConversionCastOp>(
        op, op.getOutput().getType(), op.getInput());
    return mlir::success();
  }
};


// struct ApplyOpLowering : public OpRewritePattern<gblas::ApplyOp> {
//   using OpRewritePattern<gblas::ApplyOp>::OpRewritePattern;

//   LogicalResult matchAndRewrite(gblas::ApplyOp op, PatternRewriter &rewriter) const override {
//     Location loc = op.getLoc();
//     auto resultType = llvm::cast<RankedTensorType>(op.getResult().getType());
//     int64_t rank = resultType.getRank();
    
//     Value outTensor = getInitializedOutputTensor(rewriter, loc, resultType);
    
//     SmallVector<AffineMap> maps = { 
//         rewriter.getMultiDimIdentityMap(rank), // Input
//         rewriter.getMultiDimIdentityMap(rank)  // Outs
//     };

//     auto genericOp = rewriter.create<linalg::GenericOp>(
//         loc, TypeRange{resultType}, ValueRange{op.getInput()}, ValueRange{outTensor}, maps, 
//         SmallVector<utils::IteratorType>(rank, utils::IteratorType::parallel),
//         [&](OpBuilder &b, Location nestedLoc, ValueRange args) {
//             Value resultVal = buildMathOp(b, nestedLoc, op.getApplyOp(), args[0], op.getScalar());
//             b.create<linalg::YieldOp>(nestedLoc, resultVal);
//         });

//     rewriter.replaceOp(op, genericOp.getResult(0));
//     return success();
//   }
// };

struct ApplyOpLowering : public OpRewritePattern<gblas::ApplyOp> {
    using OpRewritePattern<gblas::ApplyOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(gblas::ApplyOp op, PatternRewriter &rewriter) const override {
        Location loc = op.getLoc();
        Value inputTensor = op.getOperand(0);
        
        // Determine if this is Unary or Binary based on operand count
        bool isUnary = (op.getNumOperands() == 1);

        auto inputType = cast<RankedTensorType>(inputTensor.getType());
        
        // 1. Collect runtime sizes for any dynamic dimensions
        SmallVector<Value> dynamicSizes;
        for (int64_t i = 0; i < inputType.getRank(); ++i) {
            if (inputType.isDynamicDim(i)) {
                Value dimIdx = rewriter.create<arith::ConstantIndexOp>(loc, i);
                Value dynSize = rewriter.create<tensor::DimOp>(loc, inputTensor, dimIdx);
                dynamicSizes.push_back(dynSize);
            }
        }

        // 2. Create the empty output tensor
        Value emptyTensor = rewriter.create<tensor::EmptyOp>(
            loc, inputType.getShape(), inputType.getElementType(), dynamicSizes
        );

        // FIX: Linalg Generic only iterates over tensors.
        // Even for binary operations, the second operand is a scalar.
        // Therefore, we ONLY have 1 input tensor and 1 output tensor = 2 indexing maps.
        SmallVector<AffineMap> indexingMaps(2, rewriter.getMultiDimIdentityMap(inputType.getRank()));
        SmallVector<utils::IteratorType> iteratorTypes(inputType.getRank(), utils::IteratorType::parallel);

        SmallVector<Value> linalgInputs = {inputTensor};
        
        // Capture the scalar value outside the lambda so we can use it inside
        Value scalarVal = isUnary ? nullptr : op.getOperand(1);

        auto linalgOp = rewriter.create<linalg::GenericOp>(
            loc,
            TypeRange(inputType), 
            linalgInputs,
            ValueRange{emptyTensor},
            indexingMaps,
            iteratorTypes,
            [&](OpBuilder &b, Location nestedLoc, ValueRange args) {
                Value resultVal;
                StringRef applyOp = op.getApplyOpAttr().getValue();

                if (isUnary) {
                    // --- Handle Unary Operations ---
                    if (applyOp == "abs") {
                        resultVal = b.create<math::AbsFOp>(nestedLoc, args[0]);
                    } else {
                        // Fallback to prevent null value segfaults on unimplemented ops
                        resultVal = args[0]; 
                    }
                } else {
                    // --- Handle Binary Operations ---
                    // FIX: Populate `resultVal` using `args[0]` (the tensor element) 
                    // and `scalarVal` (the captured scalar)
                    if (applyOp == "div") {
                        resultVal = b.create<arith::DivFOp>(nestedLoc, args[0], scalarVal);
                    } else if (applyOp == "second") {
                        // "second" returns the right-hand operand
                        resultVal = scalarVal;
                    } else if (applyOp == "first") {
                        // "first" returns the left-hand operand
                        resultVal = args[0];
                    } else if (applyOp == "plus") {
                        resultVal = b.create<arith::AddFOp>(nestedLoc, args[0], scalarVal);
                    } else if (applyOp == "multiplies") {
                        resultVal = b.create<arith::MulFOp>(nestedLoc, args[0], scalarVal);
                    } else {
                        // Fallback to prevent null value segfaults on unimplemented ops
                        resultVal = args[0];
                    }
                }

                // Yielding a null resultVal is what caused the core dump.
                // With the implementations and fallbacks above, this is now safe.
                b.create<linalg::YieldOp>(nestedLoc, resultVal);
            });

        rewriter.replaceOp(op, linalgOp.getResult(0));
        return success();
    }
};

/** 
struct ReduceToVectorOpLowering : public OpRewritePattern<gblas::ReduceToVectorOp> {
  using OpRewritePattern<gblas::ReduceToVectorOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(gblas::ReduceToVectorOp op, PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto resultType = llvm::cast<RankedTensorType>(op.getResult().getType());
    MLIRContext *context = rewriter.getContext();
    
    Value outTensor = getInitializedOutputTensor(rewriter, loc, resultType);

    AffineExpr d0, d1;
    bindDims(context, d0, d1);
    
    // Axis 1 (Row reduction): map (d0, d1) -> d0
    // Axis 0 (Col reduction): map (d0, d1) -> d1
    AffineExpr outExpr = (op.getAxis() == 1) ? d0 : d1;

    SmallVector<AffineMap> maps = {
        AffineMap::get(2, 0, {d0, d1}, context),
        AffineMap::get(2, 0, {outExpr}, context)
    };
    
    SmallVector<utils::IteratorType> iterTypes = {
        (op.getAxis() == 1) ? utils::IteratorType::parallel : utils::IteratorType::reduction,
        (op.getAxis() == 1) ? utils::IteratorType::reduction : utils::IteratorType::parallel
    };

    auto genericOp = rewriter.create<linalg::GenericOp>(
        loc, TypeRange{resultType}, ValueRange{op.getInput()}, ValueRange{outTensor}, maps, iterTypes,
        [&](OpBuilder &b, Location nestedLoc, ValueRange args) {
            Value current_val = args[0];
            Value accumulator = args[1];
            Value reduced = buildMathOp(b, nestedLoc, op.getReduceOp(), accumulator, current_val);
            b.create<linalg::YieldOp>(nestedLoc, reduced);
        });

    rewriter.replaceOp(op, genericOp.getResult(0));
    return success();
  }
};
**/

struct ReduceToVectorOpLowering : public OpRewritePattern<gblas::ReduceToVectorOp> {
  using OpRewritePattern<gblas::ReduceToVectorOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(gblas::ReduceToVectorOp op, PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto resultType = cast<RankedTensorType>(op.getResult().getType());
    MLIRContext *context = rewriter.getContext();
    
    // 1. Get the runtime size for the dynamic dimension.
    // If reducing axis 1 (cols), output size is dim 0 (rows). If axis 0 (rows), output size is dim 1 (cols).
    int64_t preservedDimIdx = (op.getAxis() == 1) ? 0 : 1;
    Value dimSize = rewriter.create<tensor::DimOp>(loc, op.getInput(), preservedDimIdx);

    // 2. Create the empty tensor with the dynamic size
    Value emptyTensor = rewriter.create<tensor::EmptyOp>(
        loc, resultType.getShape(), resultType.getElementType(), ValueRange{dimSize}
    );

    // 3. Initialize the accumulator memory (Required for reductions!)
    Value initVal;
    if (op.getReduceOp() == "multiplies") {
        initVal = rewriter.create<arith::ConstantOp>(loc, rewriter.getFloatAttr(resultType.getElementType(), 1.0));
    } else {
        // Default neutral element for "plus", "count", etc. is 0.0
        initVal = rewriter.create<arith::ConstantOp>(loc, rewriter.getFloatAttr(resultType.getElementType(), 0.0));
    }
    Value outTensor = rewriter.create<linalg::FillOp>(loc, initVal, emptyTensor).getResult(0);

    // --- Rest of your existing logic ---
    AffineExpr d0, d1;
    bindDims(context, d0, d1);
    
    // Axis 1 (Row reduction): map (d0, d1) -> d0
    // Axis 0 (Col reduction): map (d0, d1) -> d1
    AffineExpr outExpr = (op.getAxis() == 1) ? d0 : d1;

    SmallVector<AffineMap> maps = {
        AffineMap::get(2, 0, {d0, d1}, context),
        AffineMap::get(2, 0, {outExpr}, context)
    };
    
    SmallVector<utils::IteratorType> iterTypes = {
        (op.getAxis() == 1) ? utils::IteratorType::parallel : utils::IteratorType::reduction,
        (op.getAxis() == 1) ? utils::IteratorType::reduction : utils::IteratorType::parallel
    };

    auto genericOp = rewriter.create<linalg::GenericOp>(
        loc, TypeRange{resultType}, ValueRange{op.getInput()}, ValueRange{outTensor}, maps, iterTypes,
        [&](OpBuilder &b, Location nestedLoc, ValueRange args) {
            Value current_val = args[0];
            Value accumulator = args[1];
            Value reduced = buildMathOp(b, nestedLoc, op.getReduceOp(), accumulator, current_val);
            b.create<linalg::YieldOp>(nestedLoc, reduced);
        });

    rewriter.replaceOp(op, genericOp.getResult(0));
    return success();
  }
};


struct ReduceToScalarOpLowering : public OpRewritePattern<gblas::ReduceToScalarOp> {
  using OpRewritePattern<gblas::ReduceToScalarOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(gblas::ReduceToScalarOp op, PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    MLIRContext *context = rewriter.getContext();
    auto inputType = llvm::cast<RankedTensorType>(op.getInput().getType());
    auto floatType = inputType.getElementType();

    // 1. Create a 0D tensor (scalar tensor)
    Value zero = rewriter.create<arith::ConstantOp>(loc, rewriter.getZeroAttr(floatType));
    Value empty0D = rewriter.create<tensor::EmptyOp>(loc, ArrayRef<int64_t>{}, floatType);
    Value init0D = rewriter.create<linalg::FillOp>(loc, zero, empty0D).getResult(0);

    // 2. Reduce 1D -> 0D
    AffineExpr d0; bindDims(context, d0);
    SmallVector<AffineMap> maps = {
        AffineMap::get(1, 0, {d0}, context), // Input (1D)
        AffineMap::get(1, 0, {}, context)    // Output (0D)
    };

    auto reduceOp = rewriter.create<linalg::GenericOp>(
        loc, RankedTensorType::get({}, floatType), ValueRange{op.getInput()}, ValueRange{init0D}, maps,
        SmallVector<utils::IteratorType>{utils::IteratorType::reduction},
        [&](OpBuilder &b, Location nestedLoc, ValueRange args) {
            Value reduced = buildMathOp(b, nestedLoc, op.getReduceOp(), args[1], args[0]);
            b.create<linalg::YieldOp>(nestedLoc, reduced);
        });

    // 3. Extract the float from the 0D tensor
    Value finalScalar = rewriter.create<tensor::ExtractOp>(loc, reduceOp.getResult(0), ValueRange{});
    rewriter.replaceOp(op, finalScalar);
    
    return success();
  }
};



// Pass Registration
struct ConvertGBLASToLinalgPass : public gblas::impl::ConvertGBLASToLinalgBase<ConvertGBLASToLinalgPass> {
    void getDependentDialects(DialectRegistry &registry) const override {
        registry.insert<linalg::LinalgDialect, bufferization::BufferizationDialect,
                        scf::SCFDialect, tensor::TensorDialect, arith::ArithDialect,
                        sparse_tensor::SparseTensorDialect, math::MathDialect>();
    }

    void runOnOperation() override {
        ConversionTarget target(getContext());
        TypeConverter typeConverter;
        typeConverter.addConversion([](Type type) { return type; });

        target.addIllegalDialect<gblas::GBLASDialect>();
        target.addLegalDialect<linalg::LinalgDialect, tensor::TensorDialect, arith::ArithDialect, scf::SCFDialect, math::MathDialect>();
        target.addLegalDialect<bufferization::BufferizationDialect>();
        target.addLegalOp<bufferization::AllocTensorOp>();
        target.addLegalDialect<sparse_tensor::SparseTensorDialect>();
        target.addLegalOp<mlir::UnrealizedConversionCastOp>();

        target.addIllegalOp<gblas::EWiseAddOp>();
        target.addIllegalOp<gblas::UpdateOp>();

        RewritePatternSet patterns(&getContext());
        
        // ADDED TransposeOpLowering and IntersectOpLowering here!
        patterns.add<FromCooOpLowering, MxmOpLowering, MxvOpLowering, VxmOpLowering, VxvOpLowering,
                     TransposeOpLowering, IntersectOpLowering, ToPtrOpLowering, FromPtrOpLowering,
                     ReduceToScalarOpLowering, ReduceToVectorOpLowering, ApplyOpLowering>(&getContext());

        patterns.add<EWiseAddLowering, NRowsLowering, NColsLowering, UpdateOpLowering>(typeConverter, &getContext());

        if (failed(applyPartialConversion(getOperation(), target, std::move(patterns)))) {
            signalPassFailure();
        }
    }
};

} // namespace

namespace mlir {
namespace gblas {
std::unique_ptr<Pass> createConvertGBLASToLinalgPass() {
  return std::make_unique<ConvertGBLASToLinalgPass>();
}
} // namespace gblas
} // namespace mlir