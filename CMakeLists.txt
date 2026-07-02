cmake_minimum_required(VERSION 3.20)

project(mlir_encoder_multiversioner LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(LLVM REQUIRED CONFIG)
find_package(MLIR REQUIRED CONFIG)

message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")
message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")

list(APPEND CMAKE_MODULE_PATH
  "${LLVM_CMAKE_DIR}"
  "${MLIR_CMAKE_DIR}"
)

include(AddLLVM)
include(AddMLIR)
include(HandleLLVMOptions)

add_library(ShortSeqPasses MODULE
  lib/Passes.cpp
  lib/ShortSeqSpecializePass.cpp
)

target_include_directories(ShortSeqPasses
  PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
    "${LLVM_INCLUDE_DIRS}"
    "${MLIR_INCLUDE_DIRS}"
)

target_link_libraries(ShortSeqPasses
  PRIVATE
    MLIRIR
    MLIRPass
    MLIRSupport
)

# Keep the plugin path deterministic for tests and local commands.
set_target_properties(ShortSeqPasses PROPERTIES
  PREFIX ""
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
)

# Minimal lit configuration. This is intentionally explicit rather than
# depending on the in-tree LLVM test configuration.
set(SHORTSEQ_BUILD_DIR "${CMAKE_BINARY_DIR}")
set(SHORTSEQ_PLUGIN "${CMAKE_BINARY_DIR}/lib/ShortSeqPasses${CMAKE_SHARED_MODULE_SUFFIX}")
set(SHORTSEQ_MLIR_OPT "${LLVM_TOOLS_BINARY_DIR}/mlir-opt")
set(SHORTSEQ_FILECHECK "${LLVM_TOOLS_BINARY_DIR}/FileCheck")

configure_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/test/lit.site.cfg.py.in"
  "${CMAKE_CURRENT_BINARY_DIR}/test/lit.site.cfg.py"
  @ONLY
)

find_program(LLVM_LIT
  NAMES llvm-lit lit
  HINTS "${LLVM_TOOLS_BINARY_DIR}"
  REQUIRED
)

add_custom_target(check-shortseq
  COMMAND "${LLVM_LIT}" -sv "${CMAKE_CURRENT_BINARY_DIR}/test"
  DEPENDS ShortSeqPasses
  USES_TERMINAL
)
