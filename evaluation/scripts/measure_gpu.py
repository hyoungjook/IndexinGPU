from evaluate import *
from constants import *

def generate_configs():
    configs = []
    # main
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
    # MT suffix / reuse_root
    for index_type in [IndexType.gpu_masstree]:
        common_config = {
            ConfigType.index_type: index_type,
            ConfigType.max_keys: DEFAULT_MAXKEY_LONG,
            ConfigType.keylen_min: DEFAULT_KEY_LENGTH,
            ConfigType.keylen_max: DEFAULT_KEY_LENGTH,
            ConfigType.num_lookups: DEFAULT_BATCH_SIZE,
            ConfigType.rep_lookup: NUM_REPEATS,
        }
        for opt_config in EXP_GPU_MASSTREE_OPTS:
            if opt_config is None:
                continue
            configs.append({**common_config, **opt_config})
    # ExtendHT hashtag
    for index_type in [IndexType.gpu_extendhashtable]:
        common_config = {
            ConfigType.index_type: index_type,
            ConfigType.max_keys: DEFAULT_MAXKEY_LONG,
            ConfigType.keylen_min: DEFAULT_KEY_LENGTH,
            ConfigType.keylen_max: DEFAULT_KEY_LENGTH,
            ConfigType.num_lookups: DEFAULT_BATCH_SIZE,
            ConfigType.rep_lookup: NUM_REPEATS,
            ConfigType.num_insdel: DEFAULT_BATCH_SIZE,
            ConfigType.rep_insdel: NUM_REPEATS,
        }
        for opt_config in EXP_GPU_EXTENDHT_OPTS:
            configs.append({**common_config, **opt_config})
    # tile size
    for index_type in [IndexType.gpu_masstree, IndexType.gpu_extendhashtable]:
        for mix_read_ratio in EXP_MIX_READ_RATIOS:
            common_config = {
                ConfigType.index_type: index_type,
                ConfigType.max_keys: DEFAULT_MAXKEY_LONG,
                ConfigType.keylen_prefix: 0,
                ConfigType.keylen_min: DEFAULT_KEY_LENGTH,
                ConfigType.keylen_max: DEFAULT_KEY_LENGTH,
                ConfigType.num_mixed: DEFAULT_BATCH_SIZE,
                ConfigType.mix_read_ratio: mix_read_ratio,
                ConfigType.rep_mixed: NUM_REPEATS
            }
            for opt_config in EXP_MIX_OPTS:
                configs.append({**common_config, **opt_config})
    ## Delete merge
    #for index_type in [IndexType.gpu_masstree, IndexType.gpu_extendhashtable]:
    #    common_config = {
    #        ConfigType.index_type: index_type,
    #        ConfigType.max_keys: DEFAULT_MAXKEY_LONG,
    #        ConfigType.keylen_prefix: 0,
    #        ConfigType.keylen_min: DEFAULT_KEY_LENGTH,
    #        ConfigType.keylen_max: DEFAULT_KEY_LENGTH,
    #        ConfigType.num_insdel: DEFAULT_BATCH_SIZE,
    #        ConfigType.rep_insdel: NUM_REPEATS,
    #    }
    #    merge_levels = [0, 2] if index_type == IndexType.gpu_masstree else [0, 1]
    #    for merge_level in merge_levels:
    #        configs.append({**common_config, OptionalConfigType.merge_level: merge_level})
    return configs

if __name__ == "__main__":
    args = parse_args_for_measure()
    configs = generate_configs()
    run_all_and_add_to_json(args, configs, "result_gpu", args.start_from)
