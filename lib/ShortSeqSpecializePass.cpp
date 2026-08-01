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
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/ErrorHandling.h"

#include <algorithm>
#include <string>

namespace shortseq {
namespace {

static constexpr int64_t kBatch = 1;
static constexpr int64_t kStageAHidden = 384;
static constexpr int64_t kStageAIntermediate = 1536;
static constexpr int64_t kStageBHidden = 64;
static constexpr int64_t kStageBIntermediate = 256;
static constexpr int64_t kE5Hidden = 384;
static constexpr int64_t kE5Intermediate = 1536;
static constexpr int64_t kE5Heads = 12;
static constexpr int64_t kE5HeadSize = 32;
static constexpr int64_t kE5MaxSequenceLength = 512;
static constexpr int64_t kDefaultLength = 16;
static constexpr unsigned kSequenceAxis = 1;
static constexpr unsigned kActivationSequenceArguments[] = {0};
static constexpr unsigned kE5SequenceArguments[] = {0, 1, 2};

enum class ContractKind { StageA, StageB, E5 };

struct SpecializationContract {
  ContractKind kind;
  llvm::StringRef name;
  int64_t hidden;
  int64_t intermediate;
  llvm::ArrayRef<unsigned> sequenceArguments;
};

static SpecializationContract getStageAContract() {
  return {ContractKind::StageA, "Stage A MLP", kStageAHidden,
          kStageAIntermediate, kActivationSequenceArguments};
}

static SpecializationContract getStageBContract() {
  return {ContractKind::StageB, "Stage B tiny encoder", kStageBHidden,
          kStageBIntermediate, kActivationSequenceArguments};
}

static SpecializationContract getE5Contract() {
  return {ContractKind::E5, "E5-small-v2", kE5Hidden, kE5Intermediate,
          kE5SequenceArguments};
}

static mlir::LogicalResult
normalizeSpecializationLengths(mlir::ModuleOp module,
                               mlir::SmallVectorImpl<int64_t> &lengths) {
  if (lengths.empty())
    lengths.push_back(kDefaultLength);

  for (int64_t length : lengths) {
    if (length <= 0)
      return module.emitError()
             << "shortseq-specialize --lengths expects positive integers, got `"
             << length << "`";
  }

  std::sort(lengths.begin(), lengths.end());
  auto duplicate = std::adjacent_find(lengths.begin(), lengths.end());
  if (duplicate != lengths.end())
    return module.emitError()
           << "shortseq-specialize --lengths contains duplicate length "
           << *duplicate;

  return mlir::success();
}

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

static mlir::LogicalResult
validateSequenceInput(mlir::func::FuncOp entry, mlir::Type type,
                      SpecializationContract contract) {
  auto tensorType = mlir::dyn_cast<mlir::RankedTensorType>(type);
  if (!tensorType)
    return entry.emitError() << "expected " << contract.name
                             << " input argument 0 to be ranked tensor<1x?x"
                             << contract.hidden << "xf32>";

  if (!tensorType.getElementType().isF32())
    return entry.emitError() << "expected " << contract.name
                             << " input argument 0 element type f32";

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

  if (tensorType.getDimSize(2) != contract.hidden)
    return entry.emitError() << "expected hidden dimension " << contract.hidden
                             << ", got " << tensorType.getDimSize(2);

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

static mlir::LogicalResult
validateE5SequenceInput(mlir::func::FuncOp entry, mlir::Type type,
                        unsigned index, llvm::StringRef name) {
  auto tensorType = mlir::dyn_cast<mlir::RankedTensorType>(type);
  if (!tensorType)
    return entry.emitError() << "expected " << name << " argument " << index
                             << " to be a ranked tensor";
  if (!tensorType.getElementType().isInteger(64))
    return entry.emitError() << "expected " << name << " argument " << index
                             << " element type i64";
  if (tensorType.getRank() != 2)
    return entry.emitError() << "expected " << name << " argument " << index
                             << " to have rank 2";
  if (tensorType.getDimSize(0) != kBatch) {
    auto diagnostic = entry.emitError()
                      << "expected " << name << " argument " << index
                      << " batch dimension 1, got ";
    if (mlir::ShapedType::isDynamic(tensorType.getDimSize(0)))
      diagnostic << "dynamic";
    else
      diagnostic << tensorType.getDimSize(0);
    return diagnostic;
  }
  if (!mlir::ShapedType::isDynamic(tensorType.getDimSize(kSequenceAxis)))
    return entry.emitError()
           << "expected " << name << " argument " << index
           << " dynamic sequence dimension at axis 1";
  return mlir::success();
}

static bool isDynamicRankedTensor(mlir::Type type) {
  auto tensorType = mlir::dyn_cast<mlir::RankedTensorType>(type);
  return tensorType && !tensorType.hasStaticShape();
}

static bool isE5DynamicTensorType(mlir::Type type) {
  auto tensorType = mlir::dyn_cast<mlir::RankedTensorType>(type);
  if (!tensorType || tensorType.hasStaticShape())
    return false;

  auto shape = tensorType.getShape();
  auto elementType = tensorType.getElementType();
  if (elementType.isInteger(64))
    return shape.equals({kBatch, mlir::ShapedType::kDynamic}) ||
           shape.equals({kBatch, 1, 1, mlir::ShapedType::kDynamic}) ||
           shape.equals({kBatch, kE5Heads, mlir::ShapedType::kDynamic}) ||
           shape.equals({mlir::ShapedType::kDynamic});
  if (elementType.isInteger(1))
    return shape.equals({kBatch, mlir::ShapedType::kDynamic});
  if (!elementType.isF32())
    return false;

  return shape.equals({kE5Heads, kE5HeadSize,
                       mlir::ShapedType::kDynamic}) ||
         shape.equals({kE5Heads, mlir::ShapedType::kDynamic, kE5HeadSize}) ||
         shape.equals({kE5Heads, mlir::ShapedType::kDynamic,
                       mlir::ShapedType::kDynamic}) ||
         shape.equals({kBatch, kE5Heads, kE5HeadSize,
                       mlir::ShapedType::kDynamic}) ||
         shape.equals(
             {kBatch, kE5Heads, mlir::ShapedType::kDynamic, 1}) ||
         shape.equals({kBatch, kE5Heads, mlir::ShapedType::kDynamic,
                       kE5HeadSize}) ||
         shape.equals({kBatch, kE5Heads, mlir::ShapedType::kDynamic,
                       mlir::ShapedType::kDynamic}) ||
         shape.equals({kBatch, kE5Heads, mlir::ShapedType::kDynamic}) ||
         shape.equals({kBatch, 1, 1, mlir::ShapedType::kDynamic}) ||
         shape.equals({kBatch, mlir::ShapedType::kDynamic, kE5Heads,
                       kE5HeadSize}) ||
         shape.equals(
             {kBatch, mlir::ShapedType::kDynamic, kE5Intermediate}) ||
         shape.equals({kBatch, mlir::ShapedType::kDynamic, 1}) ||
         shape.equals({kBatch, mlir::ShapedType::kDynamic, kE5Hidden}) ||
         shape.equals({kBatch, mlir::ShapedType::kDynamic}) ||
         shape.equals({mlir::ShapedType::kDynamic, kE5Hidden});
}

static bool isContractDynamicTensorType(mlir::Type type,
                                        SpecializationContract contract) {
  if (contract.kind == ContractKind::E5)
    return isE5DynamicTensorType(type);

  auto tensorType = mlir::dyn_cast<mlir::RankedTensorType>(type);
  if (!tensorType || !tensorType.getElementType().isF32() ||
      tensorType.hasStaticShape())
    return false;

  auto shape = tensorType.getShape();
  return shape.equals({kBatch, mlir::ShapedType::kDynamic, contract.hidden}) ||
         shape.equals({mlir::ShapedType::kDynamic, contract.hidden}) ||
         shape.equals({mlir::ShapedType::kDynamic, contract.intermediate});
}

static bool isContractEmptyTensorType(mlir::Type type,
                                      SpecializationContract contract) {
  auto tensorType = mlir::dyn_cast<mlir::RankedTensorType>(type);
  if (!tensorType || !tensorType.getElementType().isF32() ||
      tensorType.hasStaticShape())
    return false;

  auto shape = tensorType.getShape();
  return shape.equals({mlir::ShapedType::kDynamic, contract.hidden}) ||
         shape.equals({mlir::ShapedType::kDynamic, contract.intermediate});
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

static bool isAllowedDynamicTensorOp(mlir::Operation *op,
                                     SpecializationContract contract) {
  llvm::StringRef name = op->getName().getStringRef();
  if (contract.kind == ContractKind::E5)
    return name == "tensor.dim" || name == "tensor.empty" ||
           name == "tensor.expand_shape" ||
           name == "tensor.collapse_shape" ||
           name == "tensor.extract_slice" || name == "linalg.fill" ||
           name == "linalg.generic" || name == "linalg.batch_matmul" ||
           name == "linalg.transpose";

  return name == "func.return" || name == "tensor.dim" ||
         name == "tensor.empty" || name == "tensor.collapse_shape" ||
         name == "linalg.fill" || name == "linalg.matmul" ||
         name == "linalg.generic" || name == "tensor.expand_shape";
}

static mlir::LogicalResult
validateDynamicTensorValueTypes(mlir::func::FuncOp entry, mlir::Operation *op,
                                SpecializationContract contract) {
  for (mlir::Value operand : op->getOperands()) {
    if (isDynamicRankedTensor(operand.getType()) &&
        !isContractDynamicTensorType(operand.getType(), contract))
      return op->emitError() << "in @" << entry.getSymName()
                             << ", unsupported dynamic tensor operand type "
                             << operand.getType();
  }

  for (mlir::Value result : op->getResults()) {
    if (isDynamicRankedTensor(result.getType()) &&
        !isContractDynamicTensorType(result.getType(), contract))
      return op->emitError()
             << "in @" << entry.getSymName()
             << ", unsupported dynamic tensor result type " << result.getType();
  }

  return mlir::success();
}

static mlir::LogicalResult
validateDynamicEmpty(mlir::func::FuncOp entry, mlir::Operation *op,
                     mlir::Value entryInput, SpecializationContract contract) {
  if (op->getName().getStringRef() != "tensor.empty" ||
      op->getNumResults() != 1)
    return mlir::success();

  auto resultType =
      mlir::dyn_cast<mlir::RankedTensorType>(op->getResult(0).getType());
  if (!resultType || resultType.hasStaticShape())
    return mlir::success();

  if (!isContractEmptyTensorType(resultType, contract))
    return op->emitError() << "in @" << entry.getSymName()
                           << ", unsupported dynamic tensor.empty result type "
                           << resultType << "; expected tensor<?x"
                           << contract.hidden << "xf32> or tensor<?x"
                           << contract.intermediate << "xf32>";

  if (op->getNumOperands() != 1 ||
      !isSequenceLengthValue(op->getOperand(0), entryInput))
    return op->emitError()
           << "in @" << entry.getSymName()
           << ", dynamic tensor.empty size must be tensor.dim of entry "
              "argument 0 at axis 1";

  return mlir::success();
}

static mlir::LogicalResult
validateDynamicExpandShape(mlir::func::FuncOp entry, mlir::Operation *op,
                           mlir::Value entryInput,
                           SpecializationContract contract) {
  if (op->getName().getStringRef() != "tensor.expand_shape" ||
      op->getNumResults() != 1)
    return mlir::success();

  auto resultType =
      mlir::dyn_cast<mlir::RankedTensorType>(op->getResult(0).getType());
  if (!resultType || resultType.hasStaticShape())
    return mlir::success();

  auto expectedResultType =
      getF32TensorType(entry.getContext(),
                       {kBatch, mlir::ShapedType::kDynamic, contract.hidden});
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

static bool containsValue(llvm::ArrayRef<mlir::Value> values,
                          mlir::Value value) {
  return std::find(values.begin(), values.end(), value) != values.end();
}

static bool isE5EntrySequenceDim(mlir::Operation *op,
                                 mlir::func::FuncOp entry,
                                 SpecializationContract contract) {
  if (!op || op->getName().getStringRef() != "tensor.dim" ||
      op->getNumOperands() != 2 || op->getNumResults() != 1 ||
      !isIndexConstant(op->getOperand(1), kSequenceAxis))
    return false;

  for (unsigned argument : contract.sequenceArguments) {
    if (op->getOperand(0) == entry.getArgument(argument))
      return true;
  }
  return false;
}

static llvm::SmallVector<mlir::Value>
collectE5SequenceLengths(mlir::func::FuncOp entry,
                         SpecializationContract contract) {
  llvm::SmallVector<mlir::Value> values;
  auto add = [&](mlir::Value value) {
    if (!containsValue(values, value))
      values.push_back(value);
  };

  entry.walk([&](mlir::Operation *op) {
    if (isE5EntrySequenceDim(op, entry, contract))
      add(op->getResult(0));
  });

  // The imported position-id slice computes a clamped size and then asserts
  // that it equals the input width. Treat such asserted equalities as proofs,
  // not arbitrary arithmetic as sequence-length provenance.
  bool changed;
  do {
    changed = false;
    entry.walk([&](mlir::Operation *op) {
      if (op->getName().getStringRef() != "cf.assert" ||
          op->getNumOperands() != 1)
        return;
      auto compare = op->getOperand(0).getDefiningOp<mlir::arith::CmpIOp>();
      if (!compare ||
          compare.getPredicate() != mlir::arith::CmpIPredicate::eq)
        return;
      bool lhsKnown = containsValue(values, compare.getLhs());
      bool rhsKnown = containsValue(values, compare.getRhs());
      if (lhsKnown && !rhsKnown) {
        add(compare.getRhs());
        changed = true;
      } else if (rhsKnown && !lhsKnown) {
        add(compare.getLhs());
        changed = true;
      }
    });
  } while (changed);
  return values;
}

static bool isE5ExpandShapePair(mlir::RankedTensorType source,
                                mlir::RankedTensorType result) {
  if (source.getElementType() != result.getElementType())
    return false;
  auto sourceShape = source.getShape();
  auto resultShape = result.getShape();
  if (source.getElementType().isInteger(64))
    return sourceShape.equals({kBatch, mlir::ShapedType::kDynamic}) &&
           resultShape.equals(
               {kBatch, 1, 1, mlir::ShapedType::kDynamic});
  if (!source.getElementType().isF32())
    return false;

  return (sourceShape.equals({mlir::ShapedType::kDynamic, kE5Hidden}) &&
          resultShape.equals(
              {kBatch, mlir::ShapedType::kDynamic, kE5Hidden})) ||
         (sourceShape.equals(
              {kBatch, mlir::ShapedType::kDynamic, kE5Hidden}) &&
          resultShape.equals({kBatch, mlir::ShapedType::kDynamic, kE5Heads,
                              kE5HeadSize})) ||
         (sourceShape.equals({kE5Heads, mlir::ShapedType::kDynamic,
                              mlir::ShapedType::kDynamic}) &&
          resultShape.equals({kBatch, kE5Heads,
                              mlir::ShapedType::kDynamic,
                              mlir::ShapedType::kDynamic})) ||
         (sourceShape.equals(
              {kBatch, kE5Heads, mlir::ShapedType::kDynamic}) &&
          resultShape.equals(
              {kBatch, kE5Heads, mlir::ShapedType::kDynamic, 1})) ||
         (sourceShape.equals(
              {kE5Heads, mlir::ShapedType::kDynamic, kE5HeadSize}) &&
          resultShape.equals({kBatch, kE5Heads,
                              mlir::ShapedType::kDynamic, kE5HeadSize})) ||
         (sourceShape.equals({kBatch, mlir::ShapedType::kDynamic}) &&
          resultShape.equals(
              {kBatch, mlir::ShapedType::kDynamic, 1}));
}

static bool isE5CollapseShapePair(mlir::RankedTensorType source,
                                  mlir::RankedTensorType result) {
  if (source.getElementType() != result.getElementType())
    return false;
  auto sourceShape = source.getShape();
  auto resultShape = result.getShape();
  if (source.getElementType().isInteger(64))
    return sourceShape.equals({kBatch, mlir::ShapedType::kDynamic}) &&
           resultShape.equals({mlir::ShapedType::kDynamic});
  if (!source.getElementType().isF32())
    return false;

  return (sourceShape.equals(
              {kBatch, kE5Heads, mlir::ShapedType::kDynamic, kE5HeadSize}) &&
          resultShape.equals(
              {kE5Heads, mlir::ShapedType::kDynamic, kE5HeadSize})) ||
         (sourceShape.equals({kBatch, kE5Heads, kE5HeadSize,
                              mlir::ShapedType::kDynamic}) &&
          resultShape.equals({kE5Heads, kE5HeadSize,
                              mlir::ShapedType::kDynamic})) ||
         (sourceShape.equals({kBatch, kE5Heads,
                              mlir::ShapedType::kDynamic,
                              mlir::ShapedType::kDynamic}) &&
          resultShape.equals({kE5Heads, mlir::ShapedType::kDynamic,
                              mlir::ShapedType::kDynamic})) ||
         (sourceShape.equals({kBatch, mlir::ShapedType::kDynamic, kE5Heads,
                              kE5HeadSize}) &&
          resultShape.equals(
              {kBatch, mlir::ShapedType::kDynamic, kE5Hidden}));
}

static bool hasI64ArrayAttr(mlir::Operation *op, llvm::StringRef name,
                            llvm::ArrayRef<int64_t> expected) {
  auto attr = op->getAttrOfType<mlir::DenseI64ArrayAttr>(name);
  return attr && attr.asArrayRef() == expected;
}

static bool isE5PositionSlice(mlir::Operation *op,
                              int64_t resultSequenceDimension) {
  if (op->getName().getStringRef() != "tensor.extract_slice" ||
      op->getNumOperands() != 2 || op->getNumResults() != 1)
    return false;
  auto sourceType = mlir::dyn_cast<mlir::RankedTensorType>(
      op->getOperand(0).getType());
  auto resultType = mlir::dyn_cast<mlir::RankedTensorType>(
      op->getResult(0).getType());
  return sourceType && resultType &&
         sourceType.getElementType().isInteger(64) &&
         resultType.getElementType().isInteger(64) &&
         sourceType.getShape().equals({kBatch, kE5MaxSequenceLength}) &&
         resultType.getShape().equals(
             {kBatch, resultSequenceDimension}) &&
         hasI64ArrayAttr(op, "static_offsets", {0, 0}) &&
         hasI64ArrayAttr(
             op, "static_sizes",
             {kBatch, mlir::ShapedType::kDynamic}) &&
         hasI64ArrayAttr(op, "static_strides", {1, 1});
}

static mlir::LogicalResult
validateE5ShapeOperation(mlir::func::FuncOp entry, mlir::Operation *op,
                         llvm::ArrayRef<mlir::Value> sequenceLengths,
                         SpecializationContract contract) {
  llvm::StringRef name = op->getName().getStringRef();
  if (name == "tensor.dim") {
    if (!isE5EntrySequenceDim(op, entry, contract))
      return op->emitError()
             << "in @" << entry.getSymName()
             << ", tensor.dim must read an E5 entry argument at axis 1";
    return mlir::success();
  }

  if (name == "tensor.empty") {
    auto resultType = mlir::cast<mlir::RankedTensorType>(
        op->getResult(0).getType());
    if (op->getNumOperands() != countDynamicDims(resultType))
      return op->emitError()
             << "in @" << entry.getSymName()
             << ", tensor.empty dynamic-size count does not match its result";
    for (mlir::Value size : op->getOperands()) {
      if (!containsValue(sequenceLengths, size))
        return op->emitError()
               << "in @" << entry.getSymName()
               << ", dynamic tensor.empty size is not proven equal to the E5 "
                  "entry sequence dimension";
    }
    return mlir::success();
  }

  if (name == "tensor.expand_shape") {
    auto sourceType = mlir::cast<mlir::RankedTensorType>(
        op->getOperand(0).getType());
    auto resultType = mlir::cast<mlir::RankedTensorType>(
        op->getResult(0).getType());
    if (!isE5ExpandShapePair(sourceType, resultType))
      return op->emitError()
             << "in @" << entry.getSymName()
             << ", unsupported E5 tensor.expand_shape type pair";
    if (op->getNumOperands() != 1 + countDynamicDims(resultType))
      return op->emitError()
             << "in @" << entry.getSymName()
             << ", tensor.expand_shape dynamic-size count does not match its "
                "result";
    for (mlir::Value size : op->getOperands().drop_front()) {
      if (!containsValue(sequenceLengths, size))
        return op->emitError()
               << "in @" << entry.getSymName()
               << ", dynamic tensor.expand_shape size is not proven equal to "
                  "the E5 entry sequence dimension";
    }
    return mlir::success();
  }

  if (name == "tensor.collapse_shape") {
    auto sourceType = mlir::cast<mlir::RankedTensorType>(
        op->getOperand(0).getType());
    auto resultType = mlir::cast<mlir::RankedTensorType>(
        op->getResult(0).getType());
    if (!isE5CollapseShapePair(sourceType, resultType))
      return op->emitError()
             << "in @" << entry.getSymName()
             << ", unsupported E5 tensor.collapse_shape type pair";
    return mlir::success();
  }

  if (name == "tensor.extract_slice") {
    if (!isE5PositionSlice(op, mlir::ShapedType::kDynamic))
      return op->emitError()
             << "in @" << entry.getSymName()
             << ", unsupported E5 tensor.extract_slice pattern";
    if (!containsValue(sequenceLengths, op->getOperand(1)))
      return op->emitError()
             << "in @" << entry.getSymName()
             << ", tensor.extract_slice size is not proven equal to the E5 "
                "entry sequence dimension";
  }
  return mlir::success();
}

static mlir::LogicalResult validateE5Calls(mlir::func::FuncOp entry) {
  mlir::WalkResult walkResult = entry.walk([&](mlir::Operation *op) {
    llvm::StringRef name = op->getName().getStringRef();
    if (name == "func.call_indirect") {
      op->emitError() << "in @" << entry.getSymName()
                      << ", E5-small-v2 does not support indirect calls";
      return mlir::WalkResult::interrupt();
    }
    if (name != "func.call")
      return mlir::WalkResult::advance();

    auto callee = op->getAttrOfType<mlir::FlatSymbolRefAttr>("callee");
    if (callee && callee.getValue() == entry.getSymName())
      op->emitError() << "in @" << entry.getSymName()
                      << ", E5-small-v2 does not support recursion";
    else
      op->emitError() << "in @" << entry.getSymName()
                      << ", E5-small-v2 requires one inlined function body";
    return mlir::WalkResult::interrupt();
  });
  return walkResult.wasInterrupted() ? mlir::failure() : mlir::success();
}

static mlir::LogicalResult
validateE5SpecializedBody(mlir::func::FuncOp entry,
                          SpecializationContract contract) {
  if (mlir::failed(validateE5Calls(entry)))
    return mlir::failure();
  llvm::SmallVector<mlir::Value> sequenceLengths =
      collectE5SequenceLengths(entry, contract);

  mlir::WalkResult walkResult = entry.walk([&](mlir::Operation *op) {
    if (op->getName().getStringRef() == "tensor.extract_slice") {
      if (mlir::failed(validateE5ShapeOperation(entry, op, sequenceLengths,
                                                contract)))
        return mlir::WalkResult::interrupt();
      return mlir::WalkResult::advance();
    }
    if (!hasDynamicTensorOperandOrResult(op))
      return mlir::WalkResult::advance();
    if (!isAllowedDynamicTensorOp(op, contract)) {
      op->emitError() << "in @" << entry.getSymName()
                      << ", unsupported dynamic tensor operation "
                      << op->getName().getStringRef();
      return mlir::WalkResult::interrupt();
    }
    if (mlir::failed(validateDynamicTensorValueTypes(entry, op, contract)) ||
        mlir::failed(validateE5ShapeOperation(entry, op, sequenceLengths,
                                              contract))) {
      return mlir::WalkResult::interrupt();
    }
    return mlir::WalkResult::advance();
  });
  return walkResult.wasInterrupted() ? mlir::failure() : mlir::success();
}

static mlir::LogicalResult
validateSpecializedBody(mlir::func::FuncOp entry,
                        SpecializationContract contract) {
  if (contract.kind == ContractKind::E5)
    return validateE5SpecializedBody(entry, contract);

  mlir::Value entryInput = entry.getArgument(0);

  mlir::LogicalResult result = mlir::success();
  mlir::WalkResult walkResult = entry.walk([&](mlir::Operation *op) {
    if (!hasDynamicTensorOperandOrResult(op))
      return mlir::WalkResult::advance();

    if (!isAllowedDynamicTensorOp(op, contract)) {
      result = op->emitError() << "in @" << entry.getSymName()
                               << ", unsupported dynamic tensor operation "
                               << op->getName().getStringRef();
      return mlir::WalkResult::interrupt();
    }

    if (mlir::failed(validateDynamicEmpty(entry, op, entryInput, contract)) ||
        mlir::failed(
            validateDynamicExpandShape(entry, op, entryInput, contract)) ||
        mlir::failed(validateDynamicTensorValueTypes(entry, op, contract))) {
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
  auto contract = getStageAContract();
  auto functionType = entry.getFunctionType();
  if (functionType.getNumInputs() != 5)
    return entry.emitError()
           << "expected Stage A MLP entry to have 5 arguments";

  if (functionType.getNumResults() != 1)
    return entry.emitError() << "expected Stage A MLP entry to have 1 result";

  if (mlir::failed(
          validateSequenceInput(entry, functionType.getInput(0), contract)))
    return mlir::failure();

  auto *context = entry.getContext();
  auto w1Type =
      getF32TensorType(context, {contract.hidden, contract.intermediate});
  auto b1Type = getF32TensorType(context, {contract.intermediate});
  auto w2Type =
      getF32TensorType(context, {contract.intermediate, contract.hidden});
  auto b2Type = getF32TensorType(context, {contract.hidden});
  auto resultType = getF32TensorType(
      context, {kBatch, mlir::ShapedType::kDynamic, contract.hidden});

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

static mlir::LogicalResult
validateStageBTinyEncoderEntry(mlir::func::FuncOp entry) {
  auto contract = getStageBContract();
  auto functionType = entry.getFunctionType();
  if (functionType.getNumInputs() != 11)
    return entry.emitError()
           << "expected Stage B tiny encoder entry to have 11 arguments";

  if (functionType.getNumResults() != 1)
    return entry.emitError()
           << "expected Stage B tiny encoder entry to have 1 result";

  if (mlir::failed(
          validateSequenceInput(entry, functionType.getInput(0), contract)))
    return mlir::failure();

  auto *context = entry.getContext();
  auto hiddenByHidden =
      getF32TensorType(context, {contract.hidden, contract.hidden});
  auto hiddenVector = getF32TensorType(context, {contract.hidden});
  auto hiddenByIntermediate =
      getF32TensorType(context, {contract.hidden, contract.intermediate});
  auto intermediateVector = getF32TensorType(context, {contract.intermediate});
  auto intermediateByHidden =
      getF32TensorType(context, {contract.intermediate, contract.hidden});
  auto resultType = getF32TensorType(
      context, {kBatch, mlir::ShapedType::kDynamic, contract.hidden});

  if (mlir::failed(validateParameter(entry, 1, hiddenByHidden, "q_w")))
    return mlir::failure();
  if (mlir::failed(validateParameter(entry, 2, hiddenByHidden, "k_w")))
    return mlir::failure();
  if (mlir::failed(validateParameter(entry, 3, hiddenByHidden, "v_w")))
    return mlir::failure();
  if (mlir::failed(validateParameter(entry, 4, hiddenByHidden, "o_w")))
    return mlir::failure();
  if (mlir::failed(validateParameter(entry, 5, hiddenVector, "norm_scale")))
    return mlir::failure();
  if (mlir::failed(validateParameter(entry, 6, hiddenVector, "norm_bias")))
    return mlir::failure();
  if (mlir::failed(validateParameter(entry, 7, hiddenByIntermediate, "ff_w1")))
    return mlir::failure();
  if (mlir::failed(validateParameter(entry, 8, intermediateVector, "ff_b1")))
    return mlir::failure();
  if (mlir::failed(validateParameter(entry, 9, intermediateByHidden, "ff_w2")))
    return mlir::failure();
  if (mlir::failed(validateParameter(entry, 10, hiddenVector, "ff_b2")))
    return mlir::failure();

  return validateExactType(entry, functionType.getResult(0), resultType,
                           "result 0");
}

static mlir::LogicalResult validateE5Entry(mlir::func::FuncOp entry) {
  auto functionType = entry.getFunctionType();
  if (functionType.getNumInputs() != 3)
    return entry.emitError()
           << "expected E5-small-v2 entry to have 3 arguments";
  if (functionType.getNumResults() != 1)
    return entry.emitError()
           << "expected E5-small-v2 entry to have 1 result";

  constexpr llvm::StringLiteral names[] = {
      "input_ids", "attention_mask", "token_type_ids"};
  for (unsigned index = 0; index < 3; ++index) {
    if (mlir::failed(validateE5SequenceInput(
            entry, functionType.getInput(index), index, names[index])))
      return mlir::failure();
  }

  auto resultType = getF32TensorType(entry.getContext(), {kBatch, kE5Hidden});
  return validateExactType(entry, functionType.getResult(0), resultType,
                           "result 0");
}

static mlir::LogicalResult
validateE5Lengths(mlir::func::FuncOp entry,
                  llvm::ArrayRef<int64_t> lengths) {
  for (int64_t length : lengths) {
    if (length > kE5MaxSequenceLength)
      return entry.emitError()
             << "E5-small-v2 specialization length " << length
             << " exceeds the position-embedding limit "
             << kE5MaxSequenceLength;
  }
  return mlir::success();
}

static mlir::LogicalResult validateEntry(mlir::func::FuncOp entry,
                                         ContractKind kind) {
  switch (kind) {
  case ContractKind::StageA:
    return validateStageAMlpEntry(entry);
  case ContractKind::StageB:
    return validateStageBTinyEncoderEntry(entry);
  case ContractKind::E5:
    return validateE5Entry(entry);
  }
  llvm_unreachable("unknown shortseq contract");
}

static mlir::Type refineTensorType(mlir::Type type,
                                   SpecializationContract contract,
                                   int64_t sequenceLength) {
  auto tensorType = mlir::dyn_cast<mlir::RankedTensorType>(type);
  if (!tensorType)
    return type;

  auto shape = tensorType.getShape();
  if (contract.kind == ContractKind::E5) {
    if (!isE5DynamicTensorType(tensorType))
      return type;
    llvm::SmallVector<int64_t> staticShape(shape);
    for (int64_t &dimension : staticShape) {
      if (mlir::ShapedType::isDynamic(dimension))
        dimension = sequenceLength;
    }
    return tensorType.clone(staticShape);
  }

  if (!tensorType.getElementType().isF32())
    return type;

  if (shape.equals({kBatch, mlir::ShapedType::kDynamic, contract.hidden}))
    return getF32TensorType(tensorType.getContext(),
                            {kBatch, sequenceLength, contract.hidden});
  if (shape.equals({mlir::ShapedType::kDynamic, contract.hidden}))
    return getF32TensorType(tensorType.getContext(),
                            {sequenceLength, contract.hidden});
  if (shape.equals({mlir::ShapedType::kDynamic, contract.intermediate}))
    return getF32TensorType(tensorType.getContext(),
                            {sequenceLength, contract.intermediate});

  return type;
}

static mlir::FunctionType getStaticFunctionType(mlir::func::FuncOp func,
                                                SpecializationContract contract,
                                                int64_t sequenceLength) {
  llvm::SmallVector<mlir::Type> inputs;
  llvm::SmallVector<mlir::Type> results;
  for (mlir::Type input : func.getFunctionType().getInputs())
    inputs.push_back(refineTensorType(input, contract, sequenceLength));
  for (mlir::Type result : func.getFunctionType().getResults())
    results.push_back(refineTensorType(result, contract, sequenceLength));
  return mlir::FunctionType::get(func.getContext(), inputs, results);
}

static void refineValueType(mlir::Value value, SpecializationContract contract,
                            int64_t sequenceLength) {
  value.setType(refineTensorType(value.getType(), contract, sequenceLength));
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

static void refineE5ExtractSlice(mlir::Operation *op,
                                 int64_t sequenceLength) {
  if (!isE5PositionSlice(op, sequenceLength))
    return;

  mlir::SmallVector<mlir::Value> operands{op->getOperand(0)};
  op->setOperands(operands);
  op->setAttr("operandSegmentSizes",
              mlir::DenseI32ArrayAttr::get(op->getContext(), {1, 0, 0, 0}));
  op->setAttr(
      "static_sizes",
      mlir::DenseI64ArrayAttr::get(op->getContext(), {kBatch, sequenceLength}));
}

static void refineClone(mlir::func::FuncOp clone,
                        SpecializationContract contract,
                        int64_t sequenceLength) {
  clone.setType(getStaticFunctionType(clone, contract, sequenceLength));
  for (unsigned index = 0; index < clone.getNumArguments(); ++index)
    clone.getArgument(index).setType(clone.getFunctionType().getInput(index));

  clone.walk([&](mlir::Operation *op) {
    for (mlir::Value result : op->getResults())
      refineValueType(result, contract, sequenceLength);
  });

  clone.walk([&](mlir::Operation *op) {
    dropStaticEmptySizes(op);
    refineExpandShape(op);
    if (contract.kind == ContractKind::E5)
      refineE5ExtractSlice(op, sequenceLength);
  });
}

static mlir::Value emitStaticCall(mlir::OpBuilder &builder, mlir::Location loc,
                                  mlir::func::FuncOp staticFunc,
                                  mlir::ValueRange wrapperArgs,
                                  SpecializationContract contract,
                                  mlir::Type dynamicResultType) {
  auto staticFuncType = staticFunc.getFunctionType();
  mlir::Type staticResultType = staticFuncType.getResult(0);

  mlir::SmallVector<mlir::Value> staticOperands(wrapperArgs.begin(),
                                                wrapperArgs.end());
  for (unsigned argument : contract.sequenceArguments) {
    staticOperands[argument] = mlir::tensor::CastOp::create(
        builder, loc, staticFuncType.getInput(argument), wrapperArgs[argument]);
  }

  auto staticCall = mlir::func::CallOp::create(
      builder, loc, staticFunc.getSymName(), mlir::TypeRange{staticResultType},
      staticOperands);
  if (staticResultType == dynamicResultType)
    return staticCall.getResult(0);
  return mlir::tensor::CastOp::create(builder, loc, dynamicResultType,
                                      staticCall.getResult(0));
}

static mlir::Value
emitDispatchChain(mlir::OpBuilder &builder, mlir::Location loc,
                  mlir::func::FuncOp genericFunc,
                  llvm::ArrayRef<mlir::func::FuncOp> staticFuncs,
                  llvm::ArrayRef<int64_t> lengths, unsigned index,
                  llvm::ArrayRef<mlir::Value> sequenceLengthValues,
                  SpecializationContract contract,
                  mlir::ValueRange wrapperArgs) {
  auto wrapperType = genericFunc.getFunctionType();
  mlir::Value staticLengthValue =
      mlir::arith::ConstantIndexOp::create(builder, loc, lengths[index]);
  mlir::Value isStaticLength;
  for (mlir::Value sequenceLengthValue : sequenceLengthValues) {
    mlir::Value argumentMatches = mlir::arith::CmpIOp::create(
        builder, loc, mlir::arith::CmpIPredicate::eq, sequenceLengthValue,
        staticLengthValue);
    if (isStaticLength)
      isStaticLength = mlir::arith::AndIOp::create(
          builder, loc, isStaticLength, argumentMatches);
    else
      isStaticLength = argumentMatches;
  }

  auto dispatch = mlir::scf::IfOp::create(
      builder, loc, wrapperType.getResults(), isStaticLength,
      /*withElseRegion=*/true);

  builder.setInsertionPointToStart(&dispatch.getThenRegion().front());
  mlir::Value staticResult = emitStaticCall(
      builder, loc, staticFuncs[index], wrapperArgs, contract,
      wrapperType.getResult(0));
  mlir::scf::YieldOp::create(builder, loc, mlir::ValueRange{staticResult});

  builder.setInsertionPointToStart(&dispatch.getElseRegion().front());
  if (index + 1 < staticFuncs.size()) {
    mlir::Value nestedResult =
        emitDispatchChain(builder, loc, genericFunc, staticFuncs, lengths,
                          index + 1, sequenceLengthValues, contract,
                          wrapperArgs);
    mlir::scf::YieldOp::create(builder, loc, mlir::ValueRange{nestedResult});
  } else {
    auto genericCall =
        mlir::func::CallOp::create(builder, loc, genericFunc.getSymName(),
                                   wrapperType.getResults(), wrapperArgs);
    mlir::scf::YieldOp::create(builder, loc, genericCall.getResults());
  }

  builder.setInsertionPointAfter(dispatch);
  return dispatch.getResult(0);
}

static void emitDispatchWrapper(mlir::func::FuncOp genericFunc,
                                llvm::ArrayRef<mlir::func::FuncOp> staticFuncs,
                                llvm::ArrayRef<int64_t> lengths,
                                llvm::StringRef wrapperName,
                                SpecializationContract contract) {
  mlir::OpBuilder builder(genericFunc.getContext());
  builder.setInsertionPointAfter(staticFuncs.back());
  auto loc = genericFunc.getLoc();
  auto wrapperType = genericFunc.getFunctionType();

  auto wrapper =
      mlir::func::FuncOp::create(builder, loc, wrapperName, wrapperType);
  mlir::StringAttr visibility = genericFunc.getSymVisibilityAttr();
  if (visibility)
    wrapper.setSymVisibilityAttr(visibility);

  mlir::Block *body = wrapper.addEntryBlock();
  builder.setInsertionPointToStart(body);

  mlir::Value sequenceAxis =
      mlir::arith::ConstantIndexOp::create(builder, loc, kSequenceAxis);
  mlir::SmallVector<mlir::Value> sequenceLengthValues;
  for (unsigned argument : contract.sequenceArguments) {
    sequenceLengthValues.push_back(mlir::tensor::DimOp::create(
        builder, loc, body->getArgument(argument), sequenceAxis));
  }

  mlir::Value result =
      emitDispatchChain(builder, loc, genericFunc, staticFuncs, lengths, 0,
                        sequenceLengthValues, contract, body->getArguments());
  mlir::func::ReturnOp::create(builder, loc, result);
}

static mlir::LogicalResult
emitSpecializations(mlir::ModuleOp module, mlir::func::FuncOp entry,
                    llvm::ArrayRef<int64_t> lengths,
                    SpecializationContract contract,
                    bool exposeStaticVariants) {
  mlir::SymbolTable symbolTable(module);
  std::string originalName = entry.getSymName().str();
  std::string genericName = originalName + "_generic";

  if (symbolTable.lookup(genericName))
    return module.emitError() << "symbol @" << genericName << " already exists";
  for (int64_t length : lengths) {
    std::string staticName = originalName + "_s" + std::to_string(length);
    if (symbolTable.lookup(staticName))
      return module.emitError()
             << "symbol @" << staticName << " already exists";
  }

  entry.setName(genericName);
  entry->removeAttr("shortseq.entry");
  entry->removeAttr("shortseq.stage_b");
  entry->removeAttr("shortseq.e5_small_v2");

  mlir::OpBuilder builder(module.getBodyRegion());
  mlir::Operation *insertAfter = entry.getOperation();
  mlir::SmallVector<mlir::func::FuncOp> staticFuncs;
  for (int64_t length : lengths) {
    builder.setInsertionPointAfter(insertAfter);
    mlir::IRMapping mapper;
    auto staticFunc = entry.clone(mapper);
    staticFunc.setName(originalName + "_s" + std::to_string(length));
    if (exposeStaticVariants)
      staticFunc.setPublic();
    else
      staticFunc.setPrivate();
    refineClone(staticFunc, contract, length);

    builder.insert(staticFunc);
    staticFuncs.push_back(staticFunc);
    insertAfter = staticFunc.getOperation();
  }

  emitDispatchWrapper(entry, staticFuncs, lengths, originalName, contract);

  return mlir::success();
}

class ShortSeqSpecializePass final
    : public mlir::PassWrapper<ShortSeqSpecializePass,
                               mlir::OperationPass<mlir::ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ShortSeqSpecializePass)

  ShortSeqSpecializePass() = default;
  ShortSeqSpecializePass(const ShortSeqSpecializePass &other)
      : mlir::PassWrapper<ShortSeqSpecializePass,
                          mlir::OperationPass<mlir::ModuleOp>>() {
    lengths = other.lengths;
    exposeStaticVariants = other.exposeStaticVariants;
  }

  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::arith::ArithDialect, mlir::func::FuncDialect,
                    mlir::scf::SCFDialect, mlir::tensor::TensorDialect>();
  }

  llvm::StringRef getArgument() const final { return "shortseq-specialize"; }

  llvm::StringRef getDescription() const final {
    return "Specializes supported short-sequence inference entry points";
  }

  ListOption<int64_t> lengths{
      *this, "lengths",
      llvm::cl::desc("Comma-separated exact sequence lengths to specialize")};

  Option<bool> exposeStaticVariants{
      *this, "expose-static-variants",
      llvm::cl::desc("Expose static clones as public benchmark entries"),
      llvm::cl::init(false)};

  void runOnOperation() final {
    auto module = getOperation();

    llvm::SmallVector<int64_t> parsedLengths(lengths.begin(), lengths.end());
    if (mlir::failed(normalizeSpecializationLengths(module, parsedLengths))) {
      signalPassFailure();
      return;
    }

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

    mlir::func::FuncOp entry = entries.front();
    bool isStageB = entry->hasAttr("shortseq.stage_b");
    bool isE5 = entry->hasAttr("shortseq.e5_small_v2");
    if (isStageB && isE5) {
      entry.emitError() << "conflicting shortseq contract markers";
      signalPassFailure();
      return;
    }

    SpecializationContract contract = getStageAContract();
    if (isE5)
      contract = getE5Contract();
    else if (isStageB)
      contract = getStageBContract();

    if (mlir::failed(validateEntry(entry, contract.kind))) {
      signalPassFailure();
      return;
    }
    if (isE5 && mlir::failed(validateE5Lengths(entry, parsedLengths))) {
      signalPassFailure();
      return;
    }

    if (mlir::failed(validateSpecializedBody(entry, contract))) {
      signalPassFailure();
      return;
    }

    if (mlir::failed(emitSpecializations(module, entry, parsedLengths, contract,
                                         exposeStaticVariants))) {
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
