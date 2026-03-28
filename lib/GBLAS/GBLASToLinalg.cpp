#include "GBLAS/GBLASDialect.h"
#include "GBLAS/GBLASPasses.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Transforms/DialectConversion.h"

#include "mlir/Dialect/SparseTensor/IR/SparseTensor.h"

#include "mlir/AsmParser/AsmParser.h"


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

    // 1. Use the Parser to build the encoding! This is immune to C++ API changes.
    // We define a COO format (compressed, singleton)
    auto configAttr = mlir::parseAttribute(
        "#sparse_tensor.encoding<{ map = (d0, d1) -> (d0 : compressed, d1 : singleton) }>",
        rewriter.getContext());
    auto config = cast<sparse_tensor::SparseTensorEncodingAttr>(configAttr);

    // 2. Create the new Sparse Tensor Type
    auto sparseResultType = RankedTensorType::get(
        resultType.getShape(), resultType.getElementType(), config);

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

// Lowering for gblas.mxm 
struct MxmOpLowering : public OpRewritePattern<gblas::MxmOp> {
  using OpRewritePattern<gblas::MxmOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(gblas::MxmOp op, PatternRewriter &rewriter) const override {
    
    Location loc = op.getLoc();
    auto resultType = dyn_cast<RankedTensorType>(op.getResult().getType());
    if (!resultType) return failure();

    if (op.getCombineOp() != gblas::BinaryOp::multiplies || 
        op.getReduceOp() != gblas::BinaryOp::plus) {
      return failure(); 
    }

    Value emptyTensor = rewriter.create<tensor::EmptyOp>( loc, resultType.getShape(), resultType.getElementType());
    Value zero = rewriter.create<arith::ConstantOp>( loc, rewriter.getZeroAttr(resultType.getElementType()));
    Value filledTensor = rewriter.create<linalg::FillOp>(loc, ValueRange{zero}, ValueRange{emptyTensor}).getResult(0);

    Value matmulResult = rewriter.create<linalg::MatmulOp>( loc, resultType, ValueRange{op.getA(), op.getB()}, ValueRange{filledTensor}).getResult(0);

    rewriter.replaceOp(op, matmulResult);
    return success();
  }
};

struct ConvertGBLASToLinalgPass 
    : public gblas::impl::ConvertGBLASToLinalgBase<ConvertGBLASToLinalgPass> {
  void runOnOperation() override {
    ConversionTarget target(getContext());
    target.addIllegalDialect<gblas::GBLASDialect>();
    target.addLegalDialect<linalg::LinalgDialect, tensor::TensorDialect, 
                           arith::ArithDialect, scf::SCFDialect>();

    target.addLegalDialect<
      linalg::LinalgDialect,
      scf::SCFDialect,
      tensor::TensorDialect,
      arith::ArithDialect,
      sparse_tensor::SparseTensorDialect>();

    RewritePatternSet patterns(&getContext());
    patterns.add<FromCooOpLowering, MxmOpLowering>(&getContext());

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