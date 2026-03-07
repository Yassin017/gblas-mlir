#ifndef GBLAS_GBLASDIALECT_H
#define GBLAS_GBLASDIALECT_H

#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Bytecode/BytecodeOpInterface.h"

// Include auto-generated Enums
#include "GBLASEnums.h.inc"

// Include auto-generated Dialect
#include "GBLASDialect.h.inc"

// Include auto-generated Operations
#define GET_OP_CLASSES
#include "GBLASOps.h.inc"

#endif // GBLAS_GBLASDIALECT_H
