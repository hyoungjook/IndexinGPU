from evaluate import *
from constants import *

def generate_configs():
    configs = []
    # different table sizes for 4B
    for prefill_size in EXP_TABLE_SIZES_4B:
        for index_type in INDEX_TYPES_ROBUST + INDEX_TYPES_GPU_BASELINE:
            common_config = {
                ConfigType.index_type: index_type,
                ConfigType.num_prefill: prefill_size,
                ConfigType.keylen_prefix: 0,
                ConfigType.keylen_min: DEFAULT_KEY_LENGHT,
                ConfigType.keylen_max: DEFAULT_KEY_LENGHT,
                ConfigType.num_lookups: DEFAULT_BATCH_SIZE,
                ConfigType.num_insdel: DEFAULT_BATCH_SIZE,
                ConfigType.rep_lookup: NUM_REPEATS,
                ConfigType.rep_insdel: NUM_REPEATS,
            }
            if index_type in INDEX_TYPES_ORDERED:
                common_config[ConfigType.num_scans] = DEFAULT_BATCH_SIZE
                common_config[ConfigType.scan_count] = DEFAULT_SCAN_COUNT
                common_config[ConfigType.rep_scan] = NUM_REPEATS
            if index_type in INDEX_TYPES_ROBUST:
                common_config[OptionalConfigType.tile_size] = 16
                common_config[OptionalConfigType.lookup_concurrent] = 0
                common_config[ConfigType.num_mixed] = DEFAULT_BATCH_SIZE
                common_config[ConfigType.mix_read_ratio] = 0.5
                common_config[ConfigType.rep_mixed] = NUM_REPEATS
            if index_type in [IndexType.gpu_blink_tree]:
                common_config[OptionalConfigType.lookup_concurrent] = 0
            configs.append(common_config)
    # different table sizes for longkey
    for prefill_size in EXP_TABLE_SIZES_LONG:
        for index_type in INDEX_TYPES_ROBUST:
            common_config = {
                ConfigType.index_type: index_type,
                ConfigType.num_prefill: prefill_size,
                ConfigType.keylen_prefix: 0,
                ConfigType.keylen_min: DEFAULT_KEY_LENGHT,
                ConfigType.keylen_max: DEFAULT_KEY_LENGHT,
                ConfigType.num_lookups: DEFAULT_BATCH_SIZE,
                ConfigType.num_insdel: DEFAULT_BATCH_SIZE,
                ConfigType.rep_lookup: NUM_REPEATS,
                ConfigType.rep_insdel: NUM_REPEATS,
            }
            if index_type in INDEX_TYPES_ORDERED:
                common_config[ConfigType.num_scans] = DEFAULT_BATCH_SIZE
                common_config[ConfigType.scan_count] = DEFAULT_SCAN_COUNT
                common_config[ConfigType.rep_scan] = NUM_REPEATS
            common_config[OptionalConfigType.tile_size] = 16
            common_config[OptionalConfigType.lookup_concurrent] = 0
            common_config[ConfigType.num_mixed] = DEFAULT_BATCH_SIZE
            common_config[ConfigType.mix_read_ratio] = 0.5
            common_config[ConfigType.rep_mixed] = NUM_REPEATS
            configs.append(common_config)
    # different key lengths
    #for index_type in INDEX_TYPES_ROBUST:
    #    for key_length in EXP_KEY_LENGTHS:
    #        for prefix_length in [0, key_length - 1]:
    #            common_config = {
    #                ConfigType.index_type: index_type,
    #                ConfigType.num_keys: DEFAULT_NUM_KEYS,
    #                ConfigType.keylen_prefix: prefix_length,
    #                ConfigType.keylen_min: key_length,
    #                ConfigType.keylen_max: key_length,
    #                ConfigType.num_lookups: DEFAULT_NUM_KEYS,
    #                ConfigType.repeats_insert: 0,
    #                ConfigType.repeats_delete: 0,
    #                ConfigType.repeats_lookup: NUM_REPEATS,
    #                ConfigType.repeats_scan: 0,
    #                OptionalConfigType.tile_size: 16,
    #                OptionalConfigType.lookup_concurrent: 0,
    #            }
    #            configs.append(common_config)
    #            if index_type == IndexType.gpu_masstree:
    #                if GPU_MASSTREE_NOSUFFIX_TEST(prefix_length, key_length):
    #                    configs.append({
    #                        **common_config,
    #                        OptionalConfigType.enable_suffix: 0
    #                    })
    #            if index_type == IndexType.gpu_extendhashtable:
    #                configs.append({
    #                    **common_config,
    #                    OptionalConfigType.hash_tag_level: 0
    #                })
    return configs

if __name__ == "__main__":
    args = parse_args_for_measure()
    configs = generate_configs()
    run_all_and_add_to_json(args, configs, "result_gpu", 0)
