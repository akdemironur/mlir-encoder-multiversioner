#include "ShortSeq/Passes.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

namespace shortseq {
namespace {

class ShortSeqSpecializePass final
    : public mlir::PassWrapper<ShortSeqSpecializePass,
                               mlir::OperationPass<mlir::ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ShortSeqSpecializePass)

  llvm::StringRef getArgument() const final {
    return "shortseq-specialize";
  }

  llvm::StringRef getDescription() const final {
    return "Marks a module to prove that the ShortSeq pass plugin loaded";
  }

  void runOnOperation() final {
    auto module = getOperation();
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
