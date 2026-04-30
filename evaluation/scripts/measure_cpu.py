from evaluate import *
from constants import *

def generate_configs():
    configs = []
    # main
    for key_length in EXP_KEY_LENGTHS:
        for index_type in INDEX_TYPES_CPU_BASELINE:
            common_config = {
                ConfigType.index_type: index_type,
                ConfigType.max_keys: DEFAULT_MAXKEY_LONG,
                ConfigType.keylen_prefix: 0,
                ConfigType.keylen_min: key_length,
                ConfigType.keylen_max: key_length,
                ConfigType.valuelen_min: DEFAULT_VALUE_LENGTH_OVERVIEW,
                ConfigType.valuelen_max: DEFAULT_VALUE_LENGTH_OVERVIEW,
                ConfigType.num_lookups: DEFAULT_BATCH_SIZE,
                ConfigType.num_insdel: DEFAULT_BATCH_SIZE,
                ConfigType.rep_lookup: NUM_REPEATS,
                ConfigType.rep_insdel: NUM_REPEATS,
            }
            if index_type in IS_INDEX_TYPE_ORDERED:
                common_config[ConfigType.num_scans] = DEFAULT_SCAN_BATCH_SIZE
                common_config[ConfigType.scan_count] = DEFAULT_SCAN_COUNT
                common_config[ConfigType.rep_scan] = NUM_REPEATS
            common_config[ConfigType.num_mixed] = DEFAULT_BATCH_SIZE
            common_config[ConfigType.mix_read_ratio] = DEFAULT_MIX_READ_RATIO
            common_config[ConfigType.rep_mixed] = NUM_REPEATS
            configs.append(common_config)
    # meme
    for index_type in INDEX_TYPES_CPU_BASELINE:
        common_config = {
            ConfigType.index_type: index_type,
            ConfigType.dataset_file: MEME_DATASET_PATH,
            ConfigType.valuelen_min: DEFAULT_VALUE_LENGTH_OVERVIEW,
            ConfigType.valuelen_max: DEFAULT_VALUE_LENGTH_OVERVIEW,
            ConfigType.num_lookups: BATCH_SIZE_MEME,
            ConfigType.num_insdel: BATCH_SIZE_MEME,
            ConfigType.rep_lookup: NUM_REPEATS,
            ConfigType.rep_insdel: NUM_REPEATS,
            ConfigType.num_mixed: BATCH_SIZE_MEME,
            ConfigType.mix_read_ratio: DEFAULT_MIX_READ_RATIO,
            ConfigType.rep_mixed: NUM_REPEATS,
        }
        if index_type in IS_INDEX_TYPE_ORDERED:
            common_config[ConfigType.num_scans] = BATCH_SIZE_MEME
            common_config[ConfigType.scan_count] = DEFAULT_SCAN_COUNT
            common_config[ConfigType.rep_scan] = NUM_REPEATS
        configs.append(common_config)
    return configs

if __name__ == "__main__":
    args = parse_args_for_measure()
    configs = generate_configs()
    run_all_and_add_to_json(args, configs, "result_cpu", args.start_from)
