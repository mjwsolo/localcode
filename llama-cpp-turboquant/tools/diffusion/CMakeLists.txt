# fork-local (PATCHES.md 0005): the denoising loop is a static library linked into llama-server, which
# is the only shipped binary. The standalone CLI is a development aid; opt in with -DLLAMA_BUILD_DIFFUSION_TOOLS=ON.
set(TARGET llama-diffusion)
add_library(${TARGET} STATIC diffusion.cpp diffusion.h)
target_link_libraries(${TARGET} PUBLIC llama llama-common ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET} PUBLIC cxx_std_17)

option(LLAMA_BUILD_DIFFUSION_TOOLS "llama: build the standalone diffusion CLI/servers (dev aids, not shipped)" OFF)

if (LLAMA_BUILD_DIFFUSION_TOOLS)
    set(TARGET llama-diffusion-cli)
    add_executable(${TARGET} diffusion-cli.cpp)
    install(TARGETS ${TARGET} RUNTIME)
    target_link_libraries(${TARGET} PRIVATE llama-diffusion llama llama-common ${CMAKE_THREAD_LIBS_INIT})
    target_compile_features(${TARGET} PRIVATE cxx_std_17)
endif()
