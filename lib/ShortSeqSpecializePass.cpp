#include "ShortSeq/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"

#include <string>

namespace shortseq {
namespace {

static constexpr int64_t kBatch = 1;
static constexpr int64_t kHidden = 384;
static constexpr int64_t kIntermediate = 1536;
static constexpr int64_t kInitialSpecializationLength = 16;
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
    return entry.emitError() << "expected input argument 0 element type f32";

  if (tensorType.getRank() != 3)
    return entry.emitError() << "expected input argument 0 to have rank 3";

  if (countDynamicDims(tensorType) != 1)
    return entry.emitError()
           << "shortseq-specialize requires exactly one dynamic tensor "
              "dimension on the entry input";

  if (tensorType.getDimSize(0) != kBatch)
    return entry.emitError() << "expected static batch dimension 1, got "
                             << tensorType.getDimSize(0);

  if (!mlir::ShapedType::isDynamic(tensorType.getDimSize(kSequenceAxis)))
    return entry.emitError() << "expected dynamic sequence dimension at axis 1";

  if (tensorType.getDimSize(2) != kHidden)
    return entry.emitError()
           << "expected hidden dimension 384, got " << tensorType.getDimSize(2);

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
    return entry.emitError() << "expected " << name << " argument " << index
                             << " to have type " << expected;
  return mlir::success();
}

static bool isDynamicRankedTensor(mlir::Type type) {
  auto tensorType = mlir::dyn_cast<mlir::RankedTensorType>(type);
  return tensorType && !tensorType.hasStaticShape();
}

static bool isStageADynamicTensorType(mlir::Type type) {
  auto tensorType = mlir::dyn_cast<mlir::RankedTensorType>(type);
  if (!tensorType || !tensorType.getElementType().isF32() ||
      tensorType.hasStaticShape())
    return false;

  auto shape = tensorType.getShape();
  return shape.equals({kBatch, mlir::ShapedType::kDynamic, kHidden}) ||
         shape.equals({mlir::ShapedType::kDynamic, kHidden}) ||
         shape.equals({mlir::ShapedType::kDynamic, kIntermediate});
}

static bool isStageAEmptyTensorType(mlir::Type type) {
  auto tensorType = mlir::dyn_cast<mlir::RankedTensorType>(type);
  if (!tensorType || !tensorType.getElementType().isF32() ||
      tensorType.hasStaticShape())
    return false;

  auto shape = tensorType.getShape();
  return shape.equals({mlir::ShapedType::kDynamic, kHidden}) ||
         shape.equals({mlir::ShapedType::kDynamic, kIntermediate});
}

static bool isIndexConstant(mlir::Value value, int64_t expected) {
  auto *definingOp = value.getDefiningOp();
  if (!definingOp || definingOp->getName().getStringRef() != "arith.constant")
    return false;

  auto intAttr = mlir::dyn_cast_if_present<mlir::IntegerAttr>(
      definingOp->getAttr("value"));
  return intAttr && intAttr.getInt() == expected;
}

static bool isEntrySequenceDim(mlir::Operation *op, mlir::Value entryInput) {
  return op && op->getName().getStringRef() == "tensor.dim" &&
         op->getNumOperands() == 2 && op->getNumResults() == 1 &&
         op->getOperand(0) == entryInput &&
         isIndexConstant(op->getOperand(1), kSequenceAxis);
}

static bool isSequenceLengthValue(mlir::Value value, mlir::Value entryInput) {
  return isEntrySequenceDim(value.getDefiningOp(), entryInput);
}

static bool hasDynamicTensorOperandOrResult(mlir::Operation *op) {
  for (mlir::Value operand : op->getOperands()) {
    if (isDynamicRankedTensor(operand.getType()))
      return true;
  }
  for (mlir::Value result : op->getResults()) {
    if (isDynamicRankedTensor(result.getType()))
      return true;
  }
  return false;
}

static bool isAllowedStageADynamicTensorOp(mlir::Operation *op) {
  llvm::StringRef name = op->getName().getStringRef();
  return name == "func.return" || name == "tensor.dim" ||
         name == "tensor.empty" || name == "tensor.collapse_shape" ||
         name == "linalg.fill" || name == "linalg.matmul" ||
         name == "linalg.generic" || name == "tensor.expand_shape";
}

static mlir::LogicalResult
validateDynamicTensorValueTypes(mlir::func::FuncOp entry, mlir::Operation *op) {
  for (mlir::Value operand : op->getOperands()) {
    if (isDynamicRankedTensor(operand.getType()) &&
        !isStageADynamicTensorType(operand.getType()))
      return op->emitError() << "in @" << entry.getSymName()
                             << ", unsupported dynamic tensor operand type "
                             << operand.getType();
  }

  for (mlir::Value result : op->getResults()) {
    if (isDynamicRankedTensor(result.getType()) &&
        !isStageADynamicTensorType(result.getType()))
      return op->emitError()
             << "in @" << entry.getSymName()
             << ", unsupported dynamic tensor result type " << result.getType();
  }

  return mlir::success();
}

static mlir::LogicalResult validateDynamicEmpty(mlir::func::FuncOp entry,
                                                mlir::Operation *op,
                                                mlir::Value entryInput) {
  if (op->getName().getStringRef() != "tensor.empty" ||
      op->getNumResults() != 1)
    return mlir::success();

  auto resultType =
      mlir::dyn_cast<mlir::RankedTensorType>(op->getResult(0).getType());
  if (!resultType || resultType.hasStaticShape())
    return mlir::success();

  if (!isStageAEmptyTensorType(resultType))
    return op->emitError()
           << "in @" << entry.getSymName()
           << ", unsupported dynamic tensor.empty result type " << resultType
           << "; expected tensor<?x384xf32> or tensor<?x1536xf32>";

  if (op->getNumOperands() != 1 ||
      !isSequenceLengthValue(op->getOperand(0), entryInput))
    return op->emitError()
           << "in @" << entry.getSymName()
           << ", dynamic tensor.empty size must be tensor.dim of entry "
              "argument 0 at axis 1";

  return mlir::success();
}

static mlir::LogicalResult validateDynamicExpandShape(mlir::func::FuncOp entry,
                                                      mlir::Operation *op,
                                                      mlir::Value entryInput) {
  if (op->getName().getStringRef() != "tensor.expand_shape" ||
      op->getNumResults() != 1)
    return mlir::success();

  auto resultType =
      mlir::dyn_cast<mlir::RankedTensorType>(op->getResult(0).getType());
  if (!resultType || resultType.hasStaticShape())
    return mlir::success();

  auto expectedResultType = getF32TensorType(
      entry.getContext(), {kBatch, mlir::ShapedType::kDynamic, kHidden});
  if (resultType != expectedResultType)
    return op->emitError()
           << "in @" << entry.getSymName()
           << ", unsupported dynamic tensor.expand_shape result type "
           << resultType;

  if (op->getNumOperands() != 2 ||
      !isSequenceLengthValue(op->getOperand(1), entryInput))
    return op->emitError()
           << "in @" << entry.getSymName()
           << ", dynamic tensor.expand_shape output shape must use "
              "tensor.dim of entry argument 0 at axis 1";

  return mlir::success();
}

static mlir::LogicalResult validateStageAMlpBody(mlir::func::FuncOp entry) {
  mlir::Value entryInput = entry.getArgument(0);

  mlir::LogicalResult result = mlir::success();
  mlir::WalkResult walkResult = entry.walk([&](mlir::Operation *op) {
    if (!hasDynamicTensorOperandOrResult(op))
      return mlir::WalkResult::advance();

    if (!isAllowedStageADynamicTensorOp(op)) {
      result = op->emitError() << "in @" << entry.getSymName()
                               << ", unsupported dynamic tensor operation "
                               << op->getName().getStringRef();
      return mlir::WalkResult::interrupt();
    }

    if (mlir::failed(validateDynamicEmpty(entry, op, entryInput)) ||
        mlir::failed(validateDynamicExpandShape(entry, op, entryInput)) ||
        mlir::failed(validateDynamicTensorValueTypes(entry, op))) {
      result = mlir::failure();
      return mlir::WalkResult::interrupt();
    }

    return mlir::WalkResult::advance();
  });

  if (walkResult.wasInterrupted())
    return mlir::failure();
  return result;
}

static mlir::LogicalResult validateStageAMlpEntry(mlir::func::FuncOp entry) {
  auto functionType = entry.getFunctionType();
  if (functionType.getNumInputs() != 5)
    return entry.emitError()
           << "expected Stage A MLP entry to have 5 arguments";

  if (functionType.getNumResults() != 1)
    return entry.emitError() << "expected Stage A MLP entry to have 1 result";

  if (mlir::failed(validateInput(entry, functionType.getInput(0))))
    return mlir::failure();

  auto *context = entry.getContext();
  auto w1Type = getF32TensorType(context, {kHidden, kIntermediate});
  auto b1Type = getF32TensorType(context, {kIntermediate});
  auto w2Type = getF32TensorType(context, {kIntermediate, kHidden});
  auto b2Type = getF32TensorType(context, {kHidden});
  auto resultType =
      getF32TensorType(context, {kBatch, mlir::ShapedType::kDynamic, kHidden});

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

static mlir::FunctionType getStaticMlpType(mlir::MLIRContext *context,
                                           int64_t sequenceLength) {
  auto xType = getF32TensorType(context, {kBatch, sequenceLength, kHidden});
  auto w1Type = getF32TensorType(context, {kHidden, kIntermediate});
  auto b1Type = getF32TensorType(context, {kIntermediate});
  auto w2Type = getF32TensorType(context, {kIntermediate, kHidden});
  auto b2Type = getF32TensorType(context, {kHidden});
  return mlir::FunctionType::get(
      context, {xType, w1Type, b1Type, w2Type, b2Type}, {xType});
}

static mlir::Type refineStageATensorType(mlir::Type type,
                                         int64_t sequenceLength) {
  auto tensorType = mlir::dyn_cast<mlir::RankedTensorType>(type);
  if (!tensorType || !tensorType.getElementType().isF32())
    return type;

  auto shape = tensorType.getShape();
  if (shape.equals({kBatch, mlir::ShapedType::kDynamic, kHidden}))
    return getF32TensorType(tensorType.getContext(),
                            {kBatch, sequenceLength, kHidden});
  if (shape.equals({mlir::ShapedType::kDynamic, kHidden}))
    return getF32TensorType(tensorType.getContext(), {sequenceLength, kHidden});
  if (shape.equals({mlir::ShapedType::kDynamic, kIntermediate}))
    return getF32TensorType(tensorType.getContext(),
                            {sequenceLength, kIntermediate});

  return type;
}

static void refineValueType(mlir::Value value, int64_t sequenceLength) {
  value.setType(refineStageATensorType(value.getType(), sequenceLength));
}

static void dropStaticEmptySizes(mlir::Operation *op) {
  if (op->getName().getStringRef() != "tensor.empty" ||
      op->getNumResults() != 1)
    return;

  auto resultType =
      mlir::dyn_cast<mlir::RankedTensorType>(op->getResult(0).getType());
  if (resultType && resultType.hasStaticShape())
    op->setOperands(mlir::ValueRange{});
}

static void refineExpandShape(mlir::Operation *op) {
  if (op->getName().getStringRef() != "tensor.expand_shape" ||
      op->getNumOperands() == 0 || op->getNumResults() != 1)
    return;

  auto resultType =
      mlir::dyn_cast<mlir::RankedTensorType>(op->getResult(0).getType());
  if (!resultType || !resultType.hasStaticShape())
    return;

  mlir::SmallVector<mlir::Value> operands{op->getOperand(0)};
  op->setOperands(operands);
  op->setAttr(
      "static_output_shape",
      mlir::DenseI64ArrayAttr::get(op->getContext(), resultType.getShape()));
}

static void refineStageAClone(mlir::func::FuncOp clone,
                              int64_t sequenceLength) {
  clone.setType(getStaticMlpType(clone.getContext(), sequenceLength));
  clone.getArgument(0).setType(clone.getFunctionType().getInput(0));

  clone.walk([&](mlir::Operation *op) {
    for (mlir::Value result : op->getResults())
      refineValueType(result, sequenceLength);
  });

  clone.walk([](mlir::Operation *op) {
    dropStaticEmptySizes(op);
    refineExpandShape(op);
  });
}

static void emitDispatchWrapper(mlir::func::FuncOp genericFunc,
                                mlir::func::FuncOp staticFunc,
                                llvm::StringRef wrapperName,
                                int64_t sequenceLength) {
  mlir::OpBuilder builder(genericFunc.getContext());
  builder.setInsertionPointAfter(staticFunc);
  auto loc = genericFunc.getLoc();
  auto wrapperType = genericFunc.getFunctionType();

  auto wrapper =
      mlir::func::FuncOp::create(builder, loc, wrapperName, wrapperType);
  mlir::StringAttr visibility = genericFunc.getSymVisibilityAttr();
  if (visibility)
    wrapper.setSymVisibilityAttr(visibility);

  mlir::Block *body = wrapper.addEntryBlock();
  builder.setInsertionPointToStart(body);

  mlir::Value input = body->getArgument(0);
  mlir::Value sequenceAxis =
      mlir::arith::ConstantIndexOp::create(builder, loc, kSequenceAxis);
  mlir::Value sequenceLengthValue =
      mlir::tensor::DimOp::create(builder, loc, input, sequenceAxis);
  mlir::Value staticLengthValue =
      mlir::arith::ConstantIndexOp::create(builder, loc, sequenceLength);
  mlir::Value isStaticLength =
      mlir::arith::CmpIOp::create(builder, loc, mlir::arith::CmpIPredicate::eq,
                                  sequenceLengthValue, staticLengthValue);

  auto staticFuncType = staticFunc.getFunctionType();
  mlir::Type dynamicResultType = wrapperType.getResult(0);
  mlir::Type staticInputType = staticFuncType.getInput(0);
  mlir::Type staticResultType = staticFuncType.getResult(0);

  auto dispatch = mlir::scf::IfOp::create(
      builder, loc, wrapperType.getResults(), isStaticLength,
      /*withElseRegion=*/true);

  builder.setInsertionPointToStart(&dispatch.getThenRegion().front());
  mlir::Value staticInput =
      mlir::tensor::CastOp::create(builder, loc, staticInputType, input);
  mlir::SmallVector<mlir::Value> staticOperands(body->getArguments().begin(),
                                                body->getArguments().end());
  staticOperands[0] = staticInput;

  auto staticCall = mlir::func::CallOp::create(
      builder, loc, staticFunc.getSymName(), mlir::TypeRange{staticResultType},
      staticOperands);
  mlir::Value dynamicResult = mlir::tensor::CastOp::create(
      builder, loc, dynamicResultType, staticCall.getResult(0));
  mlir::scf::YieldOp::create(builder, loc, mlir::ValueRange{dynamicResult});

  builder.setInsertionPointToStart(&dispatch.getElseRegion().front());
  mlir::SmallVector<mlir::Value> genericOperands(body->getArguments().begin(),
                                                 body->getArguments().end());
  auto genericCall =
      mlir::func::CallOp::create(builder, loc, genericFunc.getSymName(),
                                 wrapperType.getResults(), genericOperands);
  mlir::scf::YieldOp::create(builder, loc, genericCall.getResults());

  builder.setInsertionPointAfter(dispatch);
  mlir::func::ReturnOp::create(builder, loc, dispatch.getResults());
}

static mlir::LogicalResult emitClone(mlir::ModuleOp module,
                                     mlir::func::FuncOp entry,
                                     int64_t sequenceLength) {
  mlir::SymbolTable symbolTable(module);
  std::string originalName = entry.getSymName().str();
  std::string genericName = originalName + "_generic";
  std::string staticName = originalName + "_s" + std::to_string(sequenceLength);

  if (symbolTable.lookup(genericName))
    return module.emitError() << "symbol @" << genericName << " already exists";
  if (symbolTable.lookup(staticName))
    return module.emitError() << "symbol @" << staticName << " already exists";

  entry.setName(genericName);
  entry->removeAttr("shortseq.entry");

  mlir::OpBuilder builder(module.getBodyRegion());
  builder.setInsertionPointAfter(entry);
  mlir::IRMapping mapper;
  auto staticFunc = entry.clone(mapper);
  staticFunc.setName(staticName);
  staticFunc.setPrivate();
  refineStageAClone(staticFunc, sequenceLength);

  builder.insert(staticFunc);
  emitDispatchWrapper(entry, staticFunc, originalName, sequenceLength);

  return mlir::success();
}

class ShortSeqSpecializePass final
    : public mlir::PassWrapper<ShortSeqSpecializePass,
                               mlir::OperationPass<mlir::ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ShortSeqSpecializePass)

  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::arith::ArithDialect, mlir::func::FuncDialect,
                    mlir::scf::SCFDialect, mlir::tensor::TensorDialect>();
  }

  llvm::StringRef getArgument() const final { return "shortseq-specialize"; }

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

    if (mlir::failed(validateStageAMlpBody(entries.front()))) {
      signalPassFailure();
      return;
    }

    if (mlir::failed(
            emitClone(module, entries.front(), kInitialSpecializationLength))) {
      signalPassFailure();
      return;
    }

    if (mlir::failed(mlir::verify(module))) {
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
