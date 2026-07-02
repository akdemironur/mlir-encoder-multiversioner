#include "ShortSeq/Passes.h"

#include "mlir/Tools/Plugins/PassPlugin.h"

extern "C" ::mlir::PassPluginLibraryInfo LLVM_ATTRIBUTE_WEAK
mlirGetPassPluginInfo() {
  return {
      MLIR_PLUGIN_API_VERSION,
      "ShortSeqPasses",
      "0.1",
      []() { shortseq::registerShortSeqPasses(); },
  };
}
