#ifndef SHORTSEQ_PASSES_H
#define SHORTSEQ_PASSES_H

#include "mlir/Pass/Pass.h"

#include <memory>

namespace shortseq {

std::unique_ptr<mlir::Pass> createShortSeqSpecializePass();

void registerShortSeqPasses();

} // namespace shortseq

#endif // SHORTSEQ_PASSES_H
