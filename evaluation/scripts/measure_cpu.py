from evaluate import *

CONFIGS = [
    {
        ConfigType.index_type: IndexType.cpu_masstree,
        ConfigType.repeats_scan: 10
    }
]

if __name__ == "__main__":
    args = parse_args_for_measure()
    run_all_and_add_to_json(args, CONFIGS, "result_cpu")
