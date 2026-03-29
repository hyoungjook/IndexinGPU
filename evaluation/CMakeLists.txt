project(evaluation)

set_property(DIRECTORY PROPERTY EXCLUDE_FROM_ALL YES)

set(EVALUATION_INCLUDE_DIRS
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}/../include
  ${CMAKE_CURRENT_SOURCE_DIR}/../baselines/MVGpuBTree/include
)

add_executable(universal_bench universal_bench.cu)
set_target_properties(universal_bench PROPERTIES FOLDER "evaluation")
target_include_directories(universal_bench PRIVATE ${EVALUATION_INCLUDE_DIRS})
target_link_libraries(universal_bench PRIVATE robust_gpu_indexing)

add_executable(universal_bench_with_baseline universal_bench.cu)
set_target_properties(universal_bench_with_baseline PROPERTIES FOLDER "evaluation")
target_include_directories(universal_bench_with_baseline PRIVATE ${EVALUATION_INCLUDE_DIRS})
target_compile_definitions(universal_bench_with_baseline PRIVATE UNIVERSAL_BENCH_WITH_BASELINE)
target_link_libraries(universal_bench_with_baseline PRIVATE robust_gpu_indexing)
