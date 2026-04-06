project(evaluation)

set_property(DIRECTORY PROPERTY EXCLUDE_FROM_ALL YES)

set(EVALUATION_INCLUDE_DIRS
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}/../include
)

add_executable(universal_bench universal_bench.cu)
set_target_properties(universal_bench PROPERTIES FOLDER "evaluation")
target_include_directories(universal_bench PRIVATE ${EVALUATION_INCLUDE_DIRS})
target_link_libraries(universal_bench PRIVATE robust_gpu_indexing)

set(UNIVERSAL_BENCH_WITH_GPU_BASELINE_SOURCES
  ${CMAKE_CURRENT_SOURCE_DIR}/gpu_dycuckoo_dynamic.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/gpu_dycuckoo_dynamic_lock.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/../baselines/DyCuckoo/dynamicHash/thirdParty/cnmem.cpp
)

set(UNIVERSAL_BENCH_WITH_GPU_BASELINE_REQUIRED_PATHS
  ${CMAKE_CURRENT_SOURCE_DIR}/../baselines/MVGpuBTree/include/gpu_blink_tree.hpp
  ${UNIVERSAL_BENCH_WITH_GPU_BASELINE_SOURCES}
)

set(UNIVERSAL_BENCH_WITH_GPU_BASELINE_MISSING_PATHS "")
foreach(required_path IN LISTS UNIVERSAL_BENCH_WITH_GPU_BASELINE_REQUIRED_PATHS)
  if(NOT EXISTS "${required_path}")
    list(APPEND UNIVERSAL_BENCH_WITH_GPU_BASELINE_MISSING_PATHS "${required_path}")
  endif()
endforeach()

list(LENGTH UNIVERSAL_BENCH_WITH_GPU_BASELINE_MISSING_PATHS
  UNIVERSAL_BENCH_WITH_GPU_BASELINE_MISSING_COUNT)

if(UNIVERSAL_BENCH_WITH_GPU_BASELINE_MISSING_COUNT EQUAL 0)
  add_executable(universal_bench_with_gpu_baseline universal_bench.cu)
  set_target_properties(universal_bench_with_gpu_baseline PROPERTIES FOLDER "evaluation")
  target_include_directories(universal_bench_with_gpu_baseline PRIVATE
    ${EVALUATION_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}/../baselines/MVGpuBTree/include
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
  )
  target_sources(universal_bench_with_gpu_baseline PRIVATE
    ${UNIVERSAL_BENCH_WITH_GPU_BASELINE_SOURCES}
  )
  target_compile_definitions(universal_bench_with_gpu_baseline PRIVATE UNIVERSAL_BENCH_WITH_GPU_BASELINE)
  target_link_libraries(universal_bench_with_gpu_baseline PRIVATE robust_gpu_indexing)
else()
  string(JOIN "\n  " UNIVERSAL_BENCH_WITH_GPU_BASELINE_MISSING_PATHS_TEXT
    ${UNIVERSAL_BENCH_WITH_GPU_BASELINE_MISSING_PATHS})
  message(STATUS
    "Skipping optional target universal_bench_with_gpu_baseline because required baseline files are missing.\n"
    "Run evaluation/scripts/prepare_baselines.sh and re-configure to enable it."
  )
endif()
