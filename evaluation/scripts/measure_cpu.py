from evaluate import *
from constants import *

def generate_configs():
    configs = []
    for index_type in INDEX_TYPES_CPU_BASELINE:
        common_config = {
            ConfigType.index_type: index_type,
            ConfigType.max_keys: DEFAULT_MAXKEY_LONG,
            ConfigType.keylen_prefix: 0,
            ConfigType.keylen_min: DEFAULT_KEY_LENGHT,
            ConfigType.keylen_max: DEFAULT_KEY_LENGHT,
            ConfigType.num_lookups: DEFAULT_BATCH_SIZE,
            ConfigType.num_insdel: DEFAULT_BATCH_SIZE,
            ConfigType.rep_lookup: NUM_REPEATS,
            ConfigType.rep_insdel: NUM_REPEATS,
        }
        if index_type in IS_INDEX_TYPE_ORDERED:
            common_config[ConfigType.num_scans] = DEFAULT_BATCH_SIZE
            common_config[ConfigType.scan_count] = DEFAULT_SCAN_COUNT
            common_config[ConfigType.rep_scan] = NUM_REPEATS
        common_config[ConfigType.num_mixed] = DEFAULT_BATCH_SIZE
        common_config[ConfigType.mix_read_ratio] = DEFAULT_MIX_READ_RATIO
        common_config[ConfigType.rep_mixed] = NUM_REPEATS
        configs.append(common_config)
    return configs

if __name__ == "__main__":
    args = parse_args_for_measure()
    configs = generate_configs()
    run_all_and_add_to_json(args, configs, "result_cpu", args.start_from)
