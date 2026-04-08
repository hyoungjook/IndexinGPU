from evaluate import *
from constants import *

def generate_configs():
    configs = []
    for key_length in EXP_KEY_LENGTHS:
        for index_type in INDEX_TYPES_ROBUST + INDEX_TYPES_GPU_BASELINE:
            if key_length > 1 and index_type not in IS_INDEX_TYPE_SUPPORT_LONGKEY:
                continue
            common_config = {
                ConfigType.index_type: index_type,
                ConfigType.max_keys: DEFAULT_MAXKEY_LONG,
                ConfigType.keylen_prefix: 0,
                ConfigType.keylen_min: key_length,
                ConfigType.keylen_max: key_length,
                ConfigType.num_lookups: DEFAULT_BATCH_SIZE,
                ConfigType.num_insdel: DEFAULT_BATCH_SIZE,
                ConfigType.rep_lookup: NUM_REPEATS,
                ConfigType.rep_insdel: NUM_REPEATS,
            }
            if index_type in IS_INDEX_TYPE_ORDERED:
                common_config[ConfigType.num_scans] = DEFAULT_BATCH_SIZE
                common_config[ConfigType.scan_count] = DEFAULT_SCAN_COUNT
                common_config[ConfigType.rep_scan] = NUM_REPEATS
            if index_type in IS_INDEX_TYPE_SUPPORT_MIX:
                common_config[ConfigType.num_mixed] = DEFAULT_BATCH_SIZE
                common_config[ConfigType.mix_read_ratio] = DEFAULT_MIX_READ_RATIO
                common_config[ConfigType.rep_mixed] = NUM_REPEATS
            if index_type == IndexType.gpu_masstree:
                common_config[OptionalConfigType.allocator_pool_ratio] = GPU_MASSTREE_LONGKEYx400M_ALLOC_POOL_RATIO
            if index_type == IndexType.gpu_chainhashtable:
                common_config[OptionalConfigType.allocator_pool_ratio] = GPU_CHAINHT_LONGKEYx400M_ALLOC_POOL_RATIO
            if index_type == IndexType.gpu_cuckoohashtable:
                common_config[OptionalConfigType.allocator_pool_ratio] = GPU_CUCKOOHT_LONGKEYx400M_ALLOC_POOL_RATIO
            configs.append(common_config)
    return configs

if __name__ == "__main__":
    args = parse_args_for_measure()
    configs = generate_configs()
    run_all_and_add_to_json(args, configs, "result_gpu", args.start_from)
