cmake_minimum_required(VERSION 3.25)
project(frizzle LANGUAGES CXX)

# One engine source, two builds (docs/architecture/logical-overview.md):
# natively on Dawn for analysis and tests, and to WebAssembly for the browser.
# Both compile against one webgpu.h, from the Dawn release in cmake/dawn.json.

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Debug CACHE STRING "" FORCE)
endif()

# The toolchain is pinned (docs/process/MEMORY.md). Any other compiler is
# refused rather than substituted.
set(FRIZZLE_EMSCRIPTEN_VERSION 6.0.8)
set(FRIZZLE_NATIVE_CLANG_VERSION 21.1.8)
if(EMSCRIPTEN)
    if(NOT EMSCRIPTEN_VERSION VERSION_EQUAL FRIZZLE_EMSCRIPTEN_VERSION)
        message(FATAL_ERROR
            "frizzle: Emscripten ${FRIZZLE_EMSCRIPTEN_VERSION} is required, found "
            "'${EMSCRIPTEN_VERSION}'. Build in the emscripten/emsdk:${FRIZZLE_EMSCRIPTEN_VERSION} image.")
    endif()
else()
    if(NOT CMAKE_CXX_COMPILER_ID STREQUAL "Clang"
       OR NOT CMAKE_CXX_COMPILER_VERSION VERSION_EQUAL FRIZZLE_NATIVE_CLANG_VERSION)
        message(FATAL_ERROR
            "frizzle: LLVM clang ${FRIZZLE_NATIVE_CLANG_VERSION} is required for the native build, "
            "found ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION} at ${CMAKE_CXX_COMPILER}. "
            "Select it with CXX=<path to clang++ ${FRIZZLE_NATIVE_CLANG_VERSION}>.")
    endif()
endif()

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(EmbedShaders)

# Diagnostic instrumentation is compiled out unless requested
# (src/core/diagnostics.h). A diagnostic build's timings are not quotable.
option(FRIZZLE_ENABLE_DIAGNOSTICS
       "Compile diagnostic instrumentation into the build (perturbs timing)" OFF)

if(NOT EMSCRIPTEN)
    # Native: analysis and tests only, on Dawn, prebuilt and pinned by
    # cmake/dawn.json. Fetch it with: python3 scripts/fetch_dawn.py
    file(READ "${CMAKE_CURRENT_SOURCE_DIR}/cmake/dawn.json" FRIZZLE_DAWN_PINS)
    string(JSON FRIZZLE_DAWN_RELEASE GET "${FRIZZLE_DAWN_PINS}" release)
    string(TOLOWER "${CMAKE_HOST_SYSTEM_PROCESSOR}" FRIZZLE_HOST_CPU)
    if(FRIZZLE_HOST_CPU STREQUAL "amd64")
        set(FRIZZLE_HOST_CPU x86_64)
    elseif(FRIZZLE_HOST_CPU STREQUAL "aarch64")
        set(FRIZZLE_HOST_CPU arm64)
    endif()
    set(FRIZZLE_DAWN_DIR
        "${CMAKE_CURRENT_SOURCE_DIR}/.cache/dawn/${FRIZZLE_DAWN_RELEASE}/${CMAKE_HOST_SYSTEM_NAME}-${FRIZZLE_HOST_CPU}")
    # lib/ on macOS, lib64/ on Linux, as Dawn's own release packages it.
    find_package(Dawn CONFIG PATHS "${FRIZZLE_DAWN_DIR}"
                 PATH_SUFFIXES lib/cmake/Dawn lib64/cmake/Dawn NO_DEFAULT_PATH)
    if(NOT Dawn_FOUND)
        message(FATAL_ERROR
            "frizzle: native Dawn ${FRIZZLE_DAWN_RELEASE} not found in ${FRIZZLE_DAWN_DIR}.\n"
            "Fetch it with: python3 scripts/fetch_dawn.py")
    endif()
    find_package(Threads REQUIRED)
endif()

# ---------------------------------------------------------------------------
# frizzle_core: the engine code that holds no WebGPU object. Compiles the same
# in both builds. Nothing here includes <emscripten.h> or anything under
# src/wasm/, tools/ or tests/ (docs/architecture/file-mapping.md).
# ---------------------------------------------------------------------------
add_library(frizzle_core STATIC
    src/core/arch/describe.cpp
    src/core/arch/gemma3/gemma3.cpp
    src/core/cache/kv.cpp
    src/core/cache/prefix.cpp
    src/core/capability/capability.cpp
    src/core/gguf/arrays.cpp
    src/core/gguf/index.cpp
    src/core/gguf/reader.cpp
    src/core/gpu/dispatch_math.cpp
    src/core/graph/graph.cpp
    src/core/kernels/attention/attention.cpp
    src/core/kernels/gather/gather.cpp
    src/core/kernels/matmul/matmul.cpp
    src/core/kernels/norm/norm.cpp
    src/core/kernels/rope/rope.cpp
    src/core/kernels/schedule.cpp
    src/core/kernels/topk/topk.cpp
    src/core/residency/file_ranges.cpp
    src/core/residency/piece_writer.cpp
    src/core/residency/plan.cpp
    src/core/residency/routes.cpp
    src/core/runtime/stops.cpp
    src/core/runtime/turn.cpp
    src/core/sampler/sampler.cpp
    src/core/tokenizer/admission.cpp
    src/core/tokenizer/bpe/merge.cpp
    src/core/tokenizer/bpe/merge_table.cpp
    src/core/tokenizer/bpe/sentencepiece_bpe.cpp
    src/core/tokenizer/bpe/sentencepiece_merges.cpp
    src/core/tokenizer/detokenizer.cpp
    src/core/tokenizer/load.cpp
    src/core/tokenizer/special.cpp
    src/core/tokenizer/unicode.cpp
    src/core/tokenizer/unicode_tables.cpp
    src/core/tokenizer/utf8_stream.cpp
    src/core/tokenizer/vocabulary.cpp
)
target_include_directories(frizzle_core PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/src")
if(FRIZZLE_ENABLE_DIAGNOSTICS)
    target_compile_definitions(frizzle_core PUBLIC FRIZZLE_DIAGNOSTICS_ENABLED=1)
    message(STATUS "frizzle: diagnostics ENABLED; timings from this build are not quotable")
endif()

frizzle_embed_shaders(
    TARGET  frizzle_core
    OUTPUT  "${CMAKE_BINARY_DIR}/generated/frizzle/shaders_generated.h"
    SHADERS "${CMAKE_CURRENT_SOURCE_DIR}/src/shaders/vector_add.wgsl"
            "${CMAKE_CURRENT_SOURCE_DIR}/src/core/formats/f32/f32.wgsl"
            "${CMAKE_CURRENT_SOURCE_DIR}/src/core/formats/f16/f16.wgsl"
            "${CMAKE_CURRENT_SOURCE_DIR}/src/core/formats/f16/f16_pack.wgsl"
            "${CMAKE_CURRENT_SOURCE_DIR}/src/core/formats/q4_0/q4_0.wgsl"
            "${CMAKE_CURRENT_SOURCE_DIR}/src/core/formats/q4_1/q4_1.wgsl"
            "${CMAKE_CURRENT_SOURCE_DIR}/src/core/formats/q8_0/q8_0.wgsl"
            "${CMAKE_CURRENT_SOURCE_DIR}/src/core/kernels/step.wgsl"
            "${CMAKE_CURRENT_SOURCE_DIR}/src/core/kernels/gather/gather.wgsl"
            "${CMAKE_CURRENT_SOURCE_DIR}/src/core/kernels/matmul/matmul.wgsl"
            "${CMAKE_CURRENT_SOURCE_DIR}/src/core/kernels/norm/norm.wgsl"
            "${CMAKE_CURRENT_SOURCE_DIR}/src/core/kernels/attention/attention.wgsl"
            "${CMAKE_CURRENT_SOURCE_DIR}/src/core/kernels/rope/rope.wgsl"
            "${CMAKE_CURRENT_SOURCE_DIR}/src/core/kernels/rope/rope_cache.wgsl"
            "${CMAKE_CURRENT_SOURCE_DIR}/src/core/kernels/rope/rope_encode.wgsl"
            "${CMAKE_CURRENT_SOURCE_DIR}/src/core/kernels/topk/topk.wgsl"
            "${CMAKE_CURRENT_SOURCE_DIR}/src/core/sampler/draw.wgsl"
)

# ---------------------------------------------------------------------------
# frizzle_gpu: the engine code that holds WebGPU objects. Natively it links
# Dawn; in the browser build, the emdawnwebgpu port of the same release.
# ---------------------------------------------------------------------------
add_library(frizzle_gpu STATIC
    src/core/gpu/device.cpp
    src/core/gpu/self_check.cpp
    src/core/kernels/program.cpp
    src/core/residency/mapping.cpp
    src/core/residency/upload.cpp
    src/core/residency/witness.cpp
    src/core/runtime/generator.cpp
    src/core/runtime/runtime.cpp
)
target_link_libraries(frizzle_gpu PUBLIC frizzle_core)
# Diagnostic only: not compiled at all in a clean build (upload_check.h).
if(FRIZZLE_ENABLE_DIAGNOSTICS)
    target_sources(frizzle_gpu PRIVATE src/core/residency/upload_check.cpp)
endif()

# Every public header that holds WebGPU objects compiles on its own, in both
# builds: one generated translation unit each.
set(FRIZZLE_GPU_HEADERS
    core/gpu/callback_mode.h
    core/gpu/device.h
    core/gpu/self_check.h
    core/gpu/userdata.h
    core/gpu/wgpu_handles.h
    core/kernels/program.h
    core/residency/mapping.h
    core/residency/upload.h
    core/residency/upload_check.h
    core/runtime/generator.h
    core/runtime/runtime.h
)
set(FRIZZLE_GPU_HEADER_CHECKS "")
foreach(header IN LISTS FRIZZLE_GPU_HEADERS)
    string(MAKE_C_IDENTIFIER "${header}" check)
    set(check "${CMAKE_BINARY_DIR}/header_checks/${check}.cpp")
    file(CONFIGURE OUTPUT "${check}" CONTENT "#include \"${header}\"\n")
    list(APPEND FRIZZLE_GPU_HEADER_CHECKS "${check}")
endforeach()
add_library(frizzle_gpu_header_checks OBJECT ${FRIZZLE_GPU_HEADER_CHECKS})
target_link_libraries(frizzle_gpu_header_checks PRIVATE frizzle_gpu)

if(EMSCRIPTEN)
    target_compile_options(frizzle_gpu PUBLIC --use-port=emdawnwebgpu)
    # The browser's event loop runs every callback (core/gpu/callback_mode.h).
    target_compile_definitions(frizzle_gpu PUBLIC
        FRIZZLE_GPU_PROCESS_EVENTS=0 FRIZZLE_GPU_DAWN_NATIVE=0)
    # The browser module and its boundary, src/wasm/, join in slice 3.
else()
    target_link_libraries(frizzle_gpu PUBLIC dawn::webgpu_dawn)
    # Callbacks run only inside wgpuInstanceProcessEvents, on the caller's
    # thread (core/gpu/callback_mode.h).
    target_compile_definitions(frizzle_gpu PUBLIC
        FRIZZLE_GPU_PROCESS_EVENTS=1 FRIZZLE_GPU_DAWN_NATIVE=1)
endif()
