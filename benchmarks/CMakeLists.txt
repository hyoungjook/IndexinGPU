project(benchmarks)

set(BENCHMARKS
    masstree_bench
    fixedhashtable_bench
    linearhashtable_bench
)

foreach(BENCHMARK ${BENCHMARKS})
    add_executable(${BENCHMARK} ${BENCHMARK}.cu)
    set_target_properties(${BENCHMARK} PROPERTIES FOLDER "benchmarks")
    target_link_libraries(${BENCHMARK} PRIVATE gpu_btrees)
endforeach()
