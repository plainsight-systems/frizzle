# The test inventory (tests/README.md), natively. Two executables:
#
#   frizzle_tests   the contract suites, registered with CTest: the routine run.
#                   Their inputs are committed; they read no model.
#   frizzle_oracle  the llama.cpp comparison, an integration test, not
#                   registered: run it when the inference path changes
#                   (docs/process/workflow.md, Tests).
#
# The oracle reads the pinned model file that scripts/fetch_model.py
# downloads; a missing file fails it, never skips it.

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

add_executable(frizzle_tests
    support/main.cpp
    contracts/edits/edit_specs_test.cpp
    contracts/gguf/model_file_test.cpp
    contracts/tokenizer/tokenizer_test.cpp
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
