add_library(libgpu__src INTERFACE)

file(GLOB_RECURSE   libgpu__src
                    "*.h"
                    "*.hpp"
                    "*.cuh"
                    "*.cu")
set(SOURCE_LIST ${libgpu__src})
target_sources(libgpu__ INTERFACE ${libgpu__src})
target_include_directories(libgpu__ INTERFACE "${CMAKE_CURRENT_SOURCE_DIR}")