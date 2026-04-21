cmake_minimum_required(VERSION 3.24 FATAL_ERROR)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES native CACHE STRING
      "CUDA architectures to build for. Defaults to the host GPU architecture(s).")
endif()

project(GpuBTree LANGUAGES CXX CUDA)

message(STATUS "CMAKE_CUDA_ARCHITECTURES=${CMAKE_CUDA_ARCHITECTURES}")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED TRUE)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)



# options
option(build_benchmarks "Build benchmarks" ON)
option(build_tests "Build tests" ON)
option(build_evaluation "Build evaluation" ON)


# Direct all output to /bin directory
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/bin)

set(json_output_directory "${CMAKE_CURRENT_SOURCE_DIR}/outputs")


# library
add_library(robust_gpu_indexing INTERFACE)

# CUDA and C++ compiler flags
set(CXX_FLAGS
        $<IF:$<CXX_COMPILER_ID:MSVC>,
        /std:c++17,
        --std=c++17
        >)
set(CUDA_FLAGS
    $<IF:$<CXX_COMPILER_ID:MSVC>,
    -Xcompiler=/std:c++17,
    -Xcompiler=-std=c++17 -Wextra>
    $<IF:$<CXX_COMPILER_ID:MSVC>,
    -Xcompiler=/W4,
    -Xcompiler=-Wall -W  -Wcast-qual -Wuninitialized>
    $<IF:$<CONFIG:Debug>,
        -G -dopt=on -src-in-ptx,
        -lineinfo>
    -maxrregcount=64
	--expt-extended-lambda
    --expt-relaxed-constexpr
    -Wno-deprecated-gpu-targets #silence deprecated architectures
    # $<$<CONFIG:Release>:-Xptxas -warn-spills -res-usage>
    -Xptxas -warn-spills
    -res-usage
    )


set(btree_includes "${CMAKE_CURRENT_SOURCE_DIR}/include")

add_library(developer_flags INTERFACE)
target_compile_options(developer_flags INTERFACE
    $<$<COMPILE_LANGUAGE:CXX>:${CXX_FLAGS}>
    $<$<COMPILE_LANGUAGE:CUDA>:${CUDA_FLAGS}>)

target_link_libraries(robust_gpu_indexing
    INTERFACE $<BUILD_INTERFACE:developer_flags>
    INTERFACE $<IF:$<CXX_COMPILER_ID:MSVC>,,-lstdc++fs>)



# Visual studio setup
set_property(GLOBAL PROPERTY USE_FOLDERS ON)
## Fix includes in visual studio for intellisense
# set_property(DIRECTORY PROPERTY VS_STARTUP_PROJECT insert_blink_custom_alloc)
# set_property(DIRECTORY PROPERTY VS_STARTUP_PROJECT insert_versioned_test)
# set_property(DIRECTORY PROPERTY VS_STARTUP_PROJECT find_versioned_test)
# set_property(DIRECTORY PROPERTY VS_STARTUP_PROJECT find_versioned_bench)
# set_property(DIRECTORY PROPERTY VS_STARTUP_PROJECT concurrent_blink_custom_alloc)
set_property(DIRECTORY PROPERTY VS_STARTUP_PROJECT concurrent_insert_rq_vtree)
# set_property(DIRECTORY PROPERTY VS_STARTUP_PROJECT reclaimation_test)
#set_property(DIRECTORY PROPERTY VS_STARTUP_PROJECT test_slab_alloc)

file(TO_NATIVE_PATH ${btree_includes} btree_includes_native)

set(CMAKE_VS_SDK_INCLUDE_DIRECTORIES
    $(VC_IncludePath)
    $(WindowsSDK_IncludePath)
    ${btree_includes_native}
    )


add_subdirectory(include)
if(build_tests)
   add_subdirectory(test)
endif(build_tests)

if(build_benchmarks)
   add_subdirectory(benchmarks)
endif(build_benchmarks)

if(build_evaluation)
   add_subdirectory(evaluation)
endif(build_evaluation)
