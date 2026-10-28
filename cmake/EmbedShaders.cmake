# frizzle_embed_shaders(TARGET <tgt> OUTPUT <header> SHADERS <files...>)
#
# Adds a build step generating <header> from the given WGSL files and makes
# <tgt> depend on it. The generated header's directory is added to <tgt>'s
# public include path.
function(frizzle_embed_shaders)
    cmake_parse_arguments(ARG "" "TARGET;OUTPUT" "SHADERS" ${ARGN})
    if(NOT ARG_TARGET OR NOT ARG_OUTPUT OR NOT ARG_SHADERS)
        message(FATAL_ERROR "frizzle_embed_shaders: TARGET, OUTPUT and SHADERS are required")
    endif()

    get_filename_component(out_dir "${ARG_OUTPUT}" DIRECTORY)
    file(MAKE_DIRECTORY "${out_dir}")

    add_custom_command(
        OUTPUT "${ARG_OUTPUT}"
        COMMAND ${CMAKE_COMMAND}
                -DOUTPUT=${ARG_OUTPUT}
                "-DSHADERS=${ARG_SHADERS}"
                -P ${CMAKE_CURRENT_FUNCTION_LIST_DIR}/embed_shaders_script.cmake
        DEPENDS ${ARG_SHADERS} ${CMAKE_CURRENT_FUNCTION_LIST_DIR}/embed_shaders_script.cmake
        COMMENT "Embedding WGSL shaders"
        VERBATIM)

    add_custom_target(${ARG_TARGET}_shaders DEPENDS "${ARG_OUTPUT}")
    add_dependencies(${ARG_TARGET} ${ARG_TARGET}_shaders)
    target_include_directories(${ARG_TARGET} PUBLIC "${out_dir}/..")
endfunction()
