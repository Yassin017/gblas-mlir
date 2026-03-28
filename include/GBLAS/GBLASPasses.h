#ifndef GBLAS_GBLASPASSES_H
#define GBLAS_GBLASPASSES_H

#include "mlir/Pass/Pass.h"
#include <memory>

namespace mlir {
namespace gblas {

// 1. MUST declare the function BEFORE the generated code tries to use it!
std::unique_ptr<Pass> createConvertGBLASToLinalgPass();

// 2. Now include the generated TableGen definitions
#define GEN_PASS_DECL
#include "GBLASPasses.h.inc"

#define GEN_PASS_REGISTRATION
#include "GBLASPasses.h.inc"

} // namespace gblas
} // namespace mlir

#endif // GBLAS_GBLASPASSES_H