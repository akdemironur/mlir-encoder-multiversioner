#include "ShortSeq/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"

namespace shortseq {
namespace {

static constexpr int64_t kBatch = 1;
static constexpr int64_t kHidden = 384;
static constexpr int64_t kIntermediate = 1536;
static constexpr unsigned kSequenceAxis = 1;

static mlir::RankedTensorType getF32TensorType(mlir::MLIRContext *context,
                                               llvm::ArrayRef<int64_t> shape) {
  return mlir::RankedTensorType::get(shape, mlir::Float32Type::get(context));
}

static unsigned countDynamicDims(mlir::RankedTensorType tensorType) {
  unsigned count = 0;
  for (int64_t dim : tensorType.getShape()) {
    if (mlir::ShapedType::isDynamic(dim))
      ++count;
  }
  return count;
}

static mlir::LogicalResult validateInput(mlir::func::FuncOp entry,
                                         mlir::Type type) {
  auto tensorType = mlir::dyn_cast<mlir::RankedTensorType>(type);
  if (!tensorType)
    return entry.emitError()
           << "expected input argument 0 to be ranked tensor<1x?x384xf32>";

  if (!tensorType.getElementType().isF32())
    return entry.emitError()
           << "expected input argument 0 element type f32";

  if (tensorType.getRank() != 3)
    return entry.emitError()
           << "expected input argument 0 to have rank 3";

  if (countDynamicDims(tensorType) != 1)
    return entry.emitError()
           << "shortseq-specialize requires exactly one dynamic tensor "
              "dimension on the entry input";

  if (tensorType.getDimSize(0) != kBatch)
    return entry.emitError()
           << "expected static batch dimension 1, got "
           << tensorType.getDimSize(0);

  if (!mlir::ShapedType::isDynamic(tensorType.getDimSize(kSequenceAxis)))
    return entry.emitError()
           << "expected dynamic sequence dimension at axis 1";

  if (tensorType.getDimSize(2) != kHidden)
    return entry.emitError()
           << "expected hidden dimension 384, got "
           << tensorType.getDimSize(2);

  return mlir::success();
}

static mlir::LogicalResult validateExactType(mlir::func::FuncOp entry,
                                             mlir::Type actual,
                                             mlir::Type expected,
                                             llvm::StringRef label) {
  if (actual != expected)
    return entry.emitError()
           << "expected " << label << " to have type " << expected;
  return mlir::success();
}

static mlir::LogicalResult validateParameter(mlir::func::FuncOp entry,
                                             unsigned index,
                                             mlir::Type expected,
                                             llvm::StringRef name) {
  auto actual = entry.getFunctionType().getInput(index);
  if (actual != expected)
    return entry.emitError()
           << "expected " << name << " argument " << index
           << " to have type " << expected;
  return mlir::success();
}

static mlir::LogicalResult validateStageAMlpEntry(mlir::func::FuncOp entry) {
  auto functionType = entry.getFunctionType();
  if (functionType.getNumInputs() != 5)
    return entry.emitError()
           << "expected Stage A MLP entry to have 5 arguments";

  if (functionType.getNumResults() != 1)
    return entry.emitError()
           << "expected Stage A MLP entry to have 1 result";

  if (mlir::failed(validateInput(entry, functionType.getInput(0))))
    return mlir::failure();

  auto *context = entry.getContext();
  auto w1Type = getF32TensorType(context, {kHidden, kIntermediate});
  auto b1Type = getF32TensorType(context, {kIntermediate});
  auto w2Type = getF32TensorType(context, {kIntermediate, kHidden});
  auto b2Type = getF32TensorType(context, {kHidden});
  auto resultType =
      getF32TensorType(context,
                       {kBatch, mlir::ShapedType::kDynamic, kHidden});

  if (mlir::failed(validateParameter(entry, 1, w1Type, "w1")))
    return mlir::failure();
  if (mlir::failed(validateParameter(entry, 2, b1Type, "b1")))
    return mlir::failure();
  if (mlir::failed(validateParameter(entry, 3, w2Type, "w2")))
    return mlir::failure();
  if (mlir::failed(validateParameter(entry, 4, b2Type, "b2")))
    return mlir::failure();

  return validateExactType(entry, functionType.getResult(0), resultType,
                           "result 0");
}

class ShortSeqSpecializePass final
    : public mlir::PassWrapper<ShortSeqSpecializePass,
                               mlir::OperationPass<mlir::ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ShortSeqSpecializePass)

  llvm::StringRef getArgument() const final {
    return "shortseq-specialize";
  }

  llvm::StringRef getDescription() const final {
    return "Validates the Stage A short-sequence MLP specialization contract";
  }

  void runOnOperation() final {
    auto module = getOperation();

    llvm::SmallVector<mlir::func::FuncOp> entries;
    for (auto func : module.getOps<mlir::func::FuncOp>()) {
      if (func->hasAttr("shortseq.entry"))
        entries.push_back(func);
    }

    if (entries.size() != 1) {
      module.emitError()
          << "shortseq-specialize requires exactly one function with "
             "shortseq.entry, found "
          << entries.size();
      signalPassFailure();
      return;
    }

    if (mlir::failed(validateStageAMlpEntry(entries.front()))) {
      signalPassFailure();
      return;
    }

    module->setAttr("shortseq.ran", mlir::UnitAttr::get(&getContext()));
  }
};

} // namespace

std::unique_ptr<mlir::Pass> createShortSeqSpecializePass() {
  return std::make_unique<ShortSeqSpecializePass>();
}

void registerShortSeqPasses() {
  mlir::PassRegistration<ShortSeqSpecializePass>();
}

} // namespace shortseq
