import lit.formats
import os

config.name = "ShortSeq"
config.test_format = lit.formats.ShTest(execute_external=True)
config.suffixes = [".mlir"]

config.test_source_root = os.path.dirname(__file__)
config.test_exec_root = os.path.join(config.shortseq_build_dir, "test-output")

config.substitutions.append(("%shortseq_plugin", config.shortseq_plugin))
config.substitutions.append(("%mlir_opt", config.shortseq_mlir_opt))
config.substitutions.append(("%FileCheck", config.shortseq_filecheck))
config.substitutions.append(("%not", config.shortseq_not))
