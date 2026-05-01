add_library(indexingpu_src INTERFACE)

file(GLOB_RECURSE   indexingpu_src
                    "*.h"
                    "*.hpp"
                    "*.cuh"
                    "*.cu")
set(SOURCE_LIST ${indexingpu_src})
target_sources(indexingpu INTERFACE ${indexingpu_src})
target_include_directories(indexingpu INTERFACE "${CMAKE_CURRENT_SOURCE_DIR}")