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

//  Lowering for gblas.from_coo 
struct FromCooOpLowering : public OpRewritePattern<gblas::FromCooOp> {
  using OpRewritePattern<gblas::FromCooOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(gblas::FromCooOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto resultType = cast<RankedTensorType>(op.getResult().getType());

    // // 1. Use the Parser to build the encoding! This is immune to C++ API changes.
    // // We define a COO format (compressed, singleton)
    // auto configAttr = mlir::parseAttribute(
    //     "#sparse_tensor.encoding<{ map = (d0, d1) -> (d0 : compressed, d1 : singleton) }>",
    //     rewriter.getContext());
    // auto config = cast<sparse_tensor::SparseTensorEncodingAttr>(configAttr);

    // // 2. Create the new Sparse Tensor Type
    // auto sparseResultType = RankedTensorType::get(
    //     resultType.getShape(), resultType.getElementType(), config);

    // Just get the exact return type the user asked for (which includes the #CSR encoding)
    auto sparseResultType = cast<RankedTensorType>(op.getResult().getType());

    // 3. FIXED: Use tensor::EmptyOp with the sparse type!
    Value sparseEmpty = rewriter.create<tensor::EmptyOp>(
        loc, sparseResultType, ValueRange{});

    // 4. The Loop
    Value c0 = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    Value c1 = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    Value length = rewriter.create<tensor::DimOp>(loc, op.getRows(), c0);

    auto forOp = rewriter.create<scf::ForOp>(
        loc, c0, length, c1, ValueRange{sparseEmpty},
        [&](OpBuilder &b, Location loc, Value iv, ValueRange iterArgs) {
            Value rowIdx = b.create<tensor::ExtractOp>(loc, op.getRows(), iv);
            Value colIdx = b.create<tensor::ExtractOp>(loc, op.getCols(), iv);
            Value val = b.create<tensor::ExtractOp>(loc, op.getVals(), iv);

            Value row = b.create<arith::IndexCastOp>(loc, b.getIndexType(), rowIdx);
            Value col = b.create<arith::IndexCastOp>(loc, b.getIndexType(), colIdx);

            Value updatedSparse = b.create<tensor::InsertOp>(
                loc, val, iterArgs[0], ValueRange{row, col});
            
            b.create<scf::YieldOp>(loc, updatedSparse);
        });

    // 1. Create the load operation to finalize the sparse tensor.
    // The "true" boolean tells MLIR that we just inserted elements, 
    // so it needs to sort and compress the memory structure now.
    mlir::Value finalizedTensor = rewriter.create<mlir::sparse_tensor::LoadOp>(
        loc, 
        forOp.getResult(0),        // The output of your scf.for loop
        true
    );

    rewriter.replaceOp(op, finalizedTensor);
    return success();
  }

};


// // Lowering for gblas.mxm 
// struct MxmOpLowering : public OpRewritePattern<gblas::MxmOp> {
//   using OpRewritePattern<gblas::MxmOp>::OpRewritePattern;

//   LogicalResult matchAndRewrite(gblas::MxmOp op, PatternRewriter &rewriter) const override {
    
//     Location loc = op.getLoc();
//     auto resultType = dyn_cast<RankedTensorType>(op.getResult().getType());
//     if (!resultType) return failure();

//     if (op.getCombineOp() != gblas::BinaryOp::multiplies || 
//         op.getReduceOp() != gblas::BinaryOp::plus) {
//       return failure(); 
//     }

//     Value emptyTensor = rewriter.create<tensor::EmptyOp>( loc, resultType.getShape(), resultType.getElementType());
//     Value zero = rewriter.create<arith::ConstantOp>( loc, rewriter.getZeroAttr(resultType.getElementType()));
//     Value filledTensor = rewriter.create<linalg::FillOp>(loc, ValueRange{zero}, ValueRange{emptyTensor}).getResult(0);

//     Value matmulResult = rewriter.create<linalg::MatmulOp>( loc, resultType, ValueRange{op.getA(), op.getB()}, ValueRange{filledTensor}).getResult(0);

//     rewriter.replaceOp(op, matmulResult);
//     return success();
//   }
// };

// // Lowering for gblas.mxm (Updated to handle optional mask rejection)
// struct MxmOpLowering : public OpRewritePattern<gblas::MxmOp> {
//   using OpRewritePattern<gblas::MxmOp>::OpRewritePattern;
//   LogicalResult matchAndRewrite(gblas::MxmOp op, PatternRewriter &rewriter) const override {
//     Location loc = op.getLoc();
//     auto resultType = dyn_cast<RankedTensorType>(op.getResult().getType());
//     if (!resultType) return failure();

//     // Reject if a mask is provided (for now, until we build mask support)
//     if (op.getMask()) return failure();

//     if (op.getCombineOp() != gblas::BinaryOp::multiplies || 
//         op.getReduceOp() != gblas::BinaryOp::plus) {
//       return failure();
//     }

//     Value emptyTensor = rewriter.create<tensor::EmptyOp>(loc, resultType.getShape(), resultType.getElementType());
//     Value zero = rewriter.create<arith::ConstantOp>(loc, rewriter.getZeroAttr(resultType.getElementType()));
//     Value filledTensor = rewriter.create<linalg::FillOp>(loc, ValueRange{zero}, ValueRange{emptyTensor}).getResult(0);

//     Value matmulResult = rewriter.create<linalg::MatmulOp>(loc, resultType, ValueRange{op.getA(), op.getB()}, ValueRange{filledTensor}).getResult(0);

//     rewriter.replaceOp(op, matmulResult);
//     return success();
//   }
// };

// Helper function to map GraphBLAS Semirings to MLIR Arith operations
static Value buildSemiringOperation(OpBuilder &builder, Location loc, gblas::BinaryOp opType, Value lhs, Value rhs) {
    // In a production compiler, you'd check the Type and emit AddI/MulI for integers.
    switch (opType) {
        case gblas::BinaryOp::plus:       return builder.create<arith::AddFOp>(loc, lhs, rhs);
        case gblas::BinaryOp::multiplies: return builder.create<arith::MulFOp>(loc, lhs, rhs);
        case gblas::BinaryOp::min:        return builder.create<arith::MinimumFOp>(loc, lhs, rhs);
        case gblas::BinaryOp::max:        return builder.create<arith::MaximumFOp>(loc, lhs, rhs);
        // Defaulting to plus if an unhandled enum is passed
        default:                          return builder.create<arith::AddFOp>(loc, lhs, rhs);
    }
}

// Lowering for gblas.mxm using linalg::GenericOp (True Semiring Support)
struct MxmOpLowering : public OpRewritePattern<gblas::MxmOp> {
  using OpRewritePattern<gblas::MxmOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(gblas::MxmOp op, PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto resultType = cast<RankedTensorType>(op.getResult().getType());

    if (op.getMask()) return failure(); // TODO: Implement structural masking later

    // 1. Create the empty output tensor and fill it with zeros
    Value emptyTensor = rewriter.create<tensor::EmptyOp>(loc, resultType.getShape(), resultType.getElementType());
    Value zero = rewriter.create<arith::ConstantOp>(loc, rewriter.getZeroAttr(resultType.getElementType()));
    Value filledTensor = rewriter.create<linalg::FillOp>(loc, ValueRange{zero}, ValueRange{emptyTensor}).getResult(0);

    // 2. Define the Affine Maps for Matmul: A(m, k) * B(k, n) = C(m, n)
    MLIRContext *context = rewriter.getContext();
    AffineExpr m, n, k;
    bindDims(context, m, n, k);
    
    SmallVector<AffineMap> indexingMaps = {
        AffineMap::get(3, 0, {m, k}, context), // Map for A
        AffineMap::get(3, 0, {k, n}, context), // Map for B
        AffineMap::get(3, 0, {m, n}, context)  // Map for C (Output)
    };

    // 3. Define the iteration types (parallel, parallel, reduction)
    SmallVector<utils::IteratorType> iteratorTypes = {
        utils::IteratorType::parallel, 
        utils::IteratorType::parallel, 
        utils::IteratorType::reduction
    };

    // 4. Create the GenericOp and inject the Semiring
    auto genericOp = rewriter.create<linalg::GenericOp>(
        loc, resultType, 
        ValueRange{op.getA(), op.getB()}, // Inputs
        ValueRange{filledTensor},         // Outputs
        indexingMaps, iteratorTypes,
        [&](OpBuilder &b, Location loc, ValueRange args) {
            Value a = args[0], b_val = args[1], c = args[2];
            
            // Apply the Combine Operator (e.g., multiplies)
            Value combined = buildSemiringOperation(b, loc, op.getCombineOp(), a, b_val);
            
            // Apply the Reduce Operator (e.g., plus)
            Value reduced = buildSemiringOperation(b, loc, op.getReduceOp(), combined, c);
            
            b.create<linalg::YieldOp>(loc, reduced);
        });

    rewriter.replaceOp(op, genericOp.getResult(0));
    return success();
  }
};

// Lowering for gblas.mxv using linalg::GenericOp
struct MxvOpLowering : public OpRewritePattern<gblas::MxvOp> {
  using OpRewritePattern<gblas::MxvOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(gblas::MxvOp op, PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto resultType = cast<RankedTensorType>(op.getResult().getType());

    if (op.getMask()) return failure();

    Value emptyTensor = rewriter.create<tensor::EmptyOp>(loc, resultType.getShape(), resultType.getElementType());
    Value zero = rewriter.create<arith::ConstantOp>(loc, rewriter.getZeroAttr(resultType.getElementType()));
    Value filledTensor = rewriter.create<linalg::FillOp>(loc, ValueRange{zero}, ValueRange{emptyTensor}).getResult(0);

    // Affine Maps for Matvec: Matrix(m, k) * Vector(k) = Output(m)
    MLIRContext *context = rewriter.getContext();
    AffineExpr m, k;
    bindDims(context, m, k);
    
    SmallVector<AffineMap> indexingMaps = {
        AffineMap::get(2, 0, {m, k}, context), // Matrix
        AffineMap::get(2, 0, {k}, context),    // Vector
        AffineMap::get(2, 0, {m}, context)     // Output
    };

    SmallVector<utils::IteratorType> iteratorTypes = {
        utils::IteratorType::parallel, 
        utils::IteratorType::reduction
    };

    auto genericOp = rewriter.create<linalg::GenericOp>(
        loc, resultType, 
        ValueRange{op.getMatrix(), op.getVector()}, 
        ValueRange{filledTensor},
        indexingMaps, iteratorTypes,
        [&](OpBuilder &b, Location loc, ValueRange args) {
            Value combined = buildSemiringOperation(b, loc, op.getCombineOp(), args[0], args[1]);
            Value reduced = buildSemiringOperation(b, loc, op.getReduceOp(), combined, args[2]);
            b.create<linalg::YieldOp>(loc, reduced);
        });

    rewriter.replaceOp(op, genericOp.getResult(0));
    return success();
  }
};

// Lowering for gblas.vxm using linalg::GenericOp
struct VxmOpLowering : public OpRewritePattern<gblas::VxmOp> {
  using OpRewritePattern<gblas::VxmOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(gblas::VxmOp op, PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto resultType = cast<RankedTensorType>(op.getResult().getType());

    if (op.getMask()) return failure(); // Reject mask for now

    Value emptyTensor = rewriter.create<tensor::EmptyOp>(loc, resultType.getShape(), resultType.getElementType());
    Value zero = rewriter.create<arith::ConstantOp>(loc, rewriter.getZeroAttr(resultType.getElementType()));
    Value filledTensor = rewriter.create<linalg::FillOp>(loc, ValueRange{zero}, ValueRange{emptyTensor}).getResult(0);

    // Affine Maps for Vector-Matrix: Vector(k) * Matrix(k, n) = Output(n)
    MLIRContext *context = rewriter.getContext();
    AffineExpr n, k;
    bindDims(context, n, k);
    
    SmallVector<AffineMap> indexingMaps = {
        AffineMap::get(2, 0, {k}, context),    // Vector
        AffineMap::get(2, 0, {k, n}, context), // Matrix
        AffineMap::get(2, 0, {n}, context)     // Output
    };

    SmallVector<utils::IteratorType> iteratorTypes = {
        utils::IteratorType::parallel,  // n
        utils::IteratorType::reduction  // k
    };

    auto genericOp = rewriter.create<linalg::GenericOp>(
        loc, resultType, 
        ValueRange{op.getVector(), op.getMatrix()}, 
        ValueRange{filledTensor},
        indexingMaps, iteratorTypes,
        [&](OpBuilder &b, Location loc, ValueRange args) {
            Value combined = buildSemiringOperation(b, loc, op.getCombineOp(), args[0], args[1]);
            Value reduced = buildSemiringOperation(b, loc, op.getReduceOp(), combined, args[2]);
            b.create<linalg::YieldOp>(loc, reduced);
        });

    rewriter.replaceOp(op, genericOp.getResult(0));
    return success();
  }
};

// Lowering for gblas.vxv using linalg::GenericOp
struct VxvOpLowering : public OpRewritePattern<gblas::VxvOp> {
  using OpRewritePattern<gblas::VxvOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(gblas::VxvOp op, PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto resultType = cast<RankedTensorType>(op.getResult().getType());

    // 0D Output Tensor (Scalar)
    Value emptyTensor = rewriter.create<tensor::EmptyOp>(loc, ArrayRef<int64_t>{}, resultType.getElementType());
    Value zero = rewriter.create<arith::ConstantOp>(loc, rewriter.getZeroAttr(resultType.getElementType()));
    Value filledTensor = rewriter.create<linalg::FillOp>(loc, ValueRange{zero}, ValueRange{emptyTensor}).getResult(0);

    // Affine Maps for Vector-Vector (Dot Product): A(k) * B(k) = Output()
    MLIRContext *context = rewriter.getContext();
    AffineExpr k;
    bindDims(context, k);
    
    SmallVector<AffineMap> indexingMaps = {
        AffineMap::get(1, 0, {k}, context), // Vector A
        AffineMap::get(1, 0, {k}, context), // Vector B
        AffineMap::get(1, 0, {}, context)   // Output (0D, so empty layout)
    };

    SmallVector<utils::IteratorType> iteratorTypes = {
        utils::IteratorType::reduction  // k
    };

    auto genericOp = rewriter.create<linalg::GenericOp>(
        loc, resultType, 
        ValueRange{op.getA(), op.getB()}, 
        ValueRange{filledTensor},
        indexingMaps, iteratorTypes,
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

  LogicalResult
  matchAndRewrite(gblas::EWiseAddOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    
    auto loc = op.getLoc();
    auto resultType = llvm::cast<RankedTensorType>(op.getResult().getType());

    // Fix: Use bufferization::AllocTensorOp instead of sparse_tensor::AllocOp
    auto allocOp = rewriter.create<bufferization::AllocTensorOp>(
        loc, resultType, ValueRange{}, Value(), IntegerAttr());

    // Replace with linalg.add
    rewriter.replaceOpWithNewOp<linalg::AddOp>(
        op, 
        resultType, 
        ValueRange{adaptor.getLhs(), adaptor.getRhs()}, 
        ValueRange{allocOp.getResult()});

    return success();
  }
};

struct NRowsLowering : public OpConversionPattern<gblas::NRowsOp> {
  using OpConversionPattern<gblas::NRowsOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(gblas::NRowsOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    Value input = adaptor.getInput();
    Type type = input.getType();

    // 1. Existing Ranked Logic
    if (auto rankedType = llvm::dyn_cast<RankedTensorType>(type)) {
      Value zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
      rewriter.replaceOpWithNewOp<tensor::DimOp>(op, input, zero);
      return success();
    }

    // 2. NEW Unranked Logic
    if (auto unrankedType = llvm::dyn_cast<UnrankedTensorType>(type)) {
      Value zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
      Value rank = rewriter.create<tensor::RankOp>(loc, input);
      
      // Check: is rank > 0?
      Value isValid = rewriter.create<arith::CmpIOp>(
          loc, arith::CmpIPredicate::ugt, rank, zero);

      Value dimSize = rewriter.create<tensor::DimOp>(loc, input, zero);
      
      // If valid, return dim, else return 0
      rewriter.replaceOpWithNewOp<arith::SelectOp>(op, isValid, dimSize, zero);
      return success();
    }
    return failure();
  }
};

// --- Updated Pattern for NColsOp ---
struct NColsLowering : public OpConversionPattern<gblas::NColsOp> {
  using OpConversionPattern<gblas::NColsOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(gblas::NColsOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    Value input = adaptor.getInput();
    Type type = input.getType();

    if (auto rankedType = llvm::dyn_cast<RankedTensorType>(type)) {
      // Safety: Only lower if rank is actually >= 2
      if (rankedType.getRank() < 2) return failure();
      Value one = rewriter.create<arith::ConstantIndexOp>(loc, 1);
      rewriter.replaceOpWithNewOp<tensor::DimOp>(op, input, one);
      return success();
    }

    if (auto unrankedType = llvm::dyn_cast<UnrankedTensorType>(type)) {
      Value zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
      Value one = rewriter.create<arith::ConstantIndexOp>(loc, 1);
      Value rank = rewriter.create<tensor::RankOp>(loc, input);

      // Check: is rank > 1?
      Value isValid = rewriter.create<arith::CmpIOp>(
          loc, arith::CmpIPredicate::ugt, rank, one);

      Value dimSize = rewriter.create<tensor::DimOp>(loc, input, one);
      
      rewriter.replaceOpWithNewOp<arith::SelectOp>(op, isValid, dimSize, zero);
      return success();
    }
    return failure();
  }
};


struct ConvertGBLASToLinalgPass 
    : public gblas::impl::ConvertGBLASToLinalgBase<ConvertGBLASToLinalgPass> {
  void runOnOperation() override {
    ConversionTarget target(getContext());
    target.addIllegalDialect<gblas::GBLASDialect>();
    target.addLegalDialect<linalg::LinalgDialect, tensor::TensorDialect, 
                           arith::ArithDialect, scf::SCFDialect>();

    target.addLegalDialect<bufferization::BufferizationDialect>();
    target.addLegalOp<bufferization::AllocTensorOp>();

    target.addLegalDialect<
      linalg::LinalgDialect,
      scf::SCFDialect,
      tensor::TensorDialect,
      arith::ArithDialect,
      sparse_tensor::SparseTensorDialect>();

    target.addIllegalOp<gblas::EWiseAddOp>();

    RewritePatternSet patterns(&getContext());

    patterns.add<FromCooOpLowering, 
                 MxmOpLowering, 
                 MxvOpLowering, 
                 VxmOpLowering, 
                 VxvOpLowering,
                 EWiseAddLowering,
                 NRowsLowering,
                 NColsLowering>(&getContext());

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



// struct MxmOpLowering : public OpRewritePattern<gblas::MxmOp> {
//   using OpRewritePattern<gblas::MxmOp>::OpRewritePattern;

//   LogicalResult matchAndRewrite(gblas::MxmOp op, PatternRewriter &rewriter) const override {
//     Location loc = op.getLoc();
//     auto resultType = dyn_cast<RankedTensorType>(op.getResult().getType());
//     if (!resultType) return failure();

//     // Use the same consistent allocator you used in EWiseAdd
//     auto allocOp = rewriter.create<bufferization::AllocTensorOp>(
//         loc, resultType, ValueRange{}, Value(), IntegerAttr());

//     // Create a zero constant for filling
//     Value zero = rewriter.create<arith::ConstantOp>(
//         loc, rewriter.getZeroAttr(resultType.getElementType()));

//     // Use linalg.fill on the allocated tensor
//     Value filledTensor = rewriter.create<linalg::FillOp>(
//         loc, ValueRange{zero}, ValueRange{allocOp}).getResult(0);

//     // Now create the Matmul
//     // Note: Use replaceOpWithNewOp to be cleaner
//     rewriter.replaceOpWithNewOp<linalg::MatmulOp>(
//         op, resultType, ValueRange{op.getA(), op.getB()}, ValueRange{filledTensor});

//     return success();
//   }
// };