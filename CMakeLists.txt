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

# The engine's libraries join here with the ported code (QUEUE, slice 1).
