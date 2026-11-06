# The test inventory (tests/README.md), natively. Four executables:
#
#   frizzle_kernels  the kernels suite: one kernel at a time on the GPU, no
#                    model; registered with CTest, so part of the routine
#                    run. It needs a GPU and fails without one, never skips.
#   frizzle_tests    the contract suites that need no device, registered with
#                    CTest: the routine run. Their inputs are committed; they
#                    read no model.
#   frizzle_oracle   the llama.cpp comparison, an integration test, not
#                    registered: run it when the inference path changes
#                    (AGENTS.md, Checks).
#   frizzle_replay   the replay and edits scenario, an integration test, not
#                    registered: run it when edits, the graph or the turn
#                    loop change.
#
# The integration tests read the pinned model file that
# scripts/fetch_model.py downloads; a missing file fails them, never skips
# them.
#
# frizzle_kernels is declared first so the compilation database's first entry
# for support/main.cpp carries Dawn's headers: clangd (.clangd) compiles the
# headers under support/ with that entry's flags, and gpu.h and
# scratch_program.h include webgpu.h. The suites do not depend on each other's
# order.

include(FetchContent)

# doctest, pinned by commit (the v2.4.12 release): a moving ref would make the
# test build a moving target.
FetchContent_Declare(
    doctest
    GIT_REPOSITORY https://github.com/doctest/doctest.git
    GIT_TAG        1da23a3e8119ec5cce4f9388e91b065e20bf06f5
)
FetchContent_MakeAvailable(doctest)

set(FRIZZLE_MODEL_FILE "${PROJECT_SOURCE_DIR}/.cache/models/gemma-3-1b-it-q4_0.gguf")

add_executable(frizzle_kernels
    support/main.cpp
    contracts/kernels/edit_kernel_test.cpp
    contracts/kernels/capture_test.cpp
    contracts/kernels/norm_capture_test.cpp
    contracts/kernels/attention_capture_test.cpp
    contracts/kernels/weights_test.cpp
)
target_link_libraries(frizzle_kernels PRIVATE frizzle_gpu doctest::doctest)
target_include_directories(frizzle_kernels PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
add_test(NAME frizzle_kernels COMMAND frizzle_kernels)

add_executable(frizzle_tests
    support/main.cpp
    contracts/cache/prefix_test.cpp
    contracts/edits/edit_specs_test.cpp
    contracts/gguf/model_file_test.cpp
    contracts/tokenizer/tokenizer_test.cpp
    contracts/turn/turn_test.cpp
)
target_link_libraries(frizzle_tests PRIVATE frizzle_core doctest::doctest)
target_include_directories(frizzle_tests PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
target_compile_definitions(frizzle_tests PRIVATE
    FRIZZLE_FIXTURE_DIR="${CMAKE_CURRENT_SOURCE_DIR}/fixtures")
add_test(NAME frizzle_tests COMMAND frizzle_tests)

add_executable(frizzle_oracle
    support/main.cpp
    integration/reference-logits/logits_test.cpp
)
target_link_libraries(frizzle_oracle PRIVATE frizzle_gpu doctest::doctest)
target_include_directories(frizzle_oracle PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
target_compile_definitions(frizzle_oracle PRIVATE
    FRIZZLE_MODEL_FILE="${FRIZZLE_MODEL_FILE}")

add_executable(frizzle_replay
    support/main.cpp
    integration/replay/replay_test.cpp
)
target_link_libraries(frizzle_replay PRIVATE frizzle_gpu doctest::doctest)
target_include_directories(frizzle_replay PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
target_compile_definitions(frizzle_replay PRIVATE
    FRIZZLE_MODEL_FILE="${FRIZZLE_MODEL_FILE}")
