from evaluate import *
from constants import *

def generate_configs():
    configs = []
    # main
    for key_length in EXP_KEY_LENGTHS:
        for index_type in INDEX_TYPES_ROBUST + INDEX_TYPES_GPU_BASELINE:
            if not DO_TEST_FOR_INDEX_TYPE(index_type, key_length):
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
                common_config[ConfigType.num_scans] = DEFAULT_SCAN_BATCH_SIZE
                common_config[ConfigType.scan_count] = DEFAULT_SCAN_COUNT
                common_config[ConfigType.rep_scan] = NUM_REPEATS
            if index_type in IS_INDEX_TYPE_SUPPORT_MIX:
                common_config[ConfigType.num_mixed] = DEFAULT_BATCH_SIZE
                common_config[ConfigType.mix_read_ratio] = DEFAULT_MIX_READ_RATIO
                common_config[ConfigType.rep_mixed] = NUM_REPEATS
            if index_type in INDEX_TYPES_ROBUST:
                common_config[OptionalConfigType.allocator_pool_ratio] = ROBUST_INDEX_ALLOC_POOL_RATIO(index_type)
            if index_type == IndexType.gpu_dycuckoo:
                common_config[OptionalConfigType.initial_capacity] = DYCUCKOO_INITIAL_CAPACITY
            configs.append(common_config)
            cpu_input_configs = {ConfigType.use_pinned_host_memory: 1}
            if index_type in INDEX_TYPES_ROBUST:
                cpu_input_configs[OptionalConfigType.use_shmem_key] = 1
            configs.append({**common_config, **cpu_input_configs})
    # MT suffix / reuse_root
    for index_type in [IndexType.gpu_masstree]:
        common_config = {
            ConfigType.index_type: index_type,
            ConfigType.max_keys: DEFAULT_MAXKEY_LONG,
            ConfigType.keylen_min: DEFAULT_KEY_LENGTH,
            ConfigType.keylen_max: DEFAULT_KEY_LENGTH,
            ConfigType.num_lookups: DEFAULT_BATCH_SIZE,
            ConfigType.rep_lookup: NUM_REPEATS,
            OptionalConfigType.allocator_pool_ratio: ROBUST_INDEX_ALLOC_POOL_RATIO(index_type)
        }
        for opt_config in EXP_GPU_MASSTREE_OPTS:
            if opt_config is None:
                continue
            configs.append({**common_config, **opt_config})
    # ExtendHT hashtag
    for index_type in [IndexType.gpu_extendhashtable]:
        common_config = {
            ConfigType.index_type: index_type,
            ConfigType.max_keys: 2 * DEFAULT_BATCH_SIZE, # use smaller max_key to see split impact
            ConfigType.keylen_min: DEFAULT_KEY_LENGTH,
            ConfigType.keylen_max: DEFAULT_KEY_LENGTH,
            ConfigType.num_lookups: DEFAULT_BATCH_SIZE,
            ConfigType.rep_lookup: NUM_REPEATS,
            ConfigType.num_insdel: DEFAULT_BATCH_SIZE,
            ConfigType.rep_insdel: NUM_REPEATS,
            OptionalConfigType.allocator_pool_ratio: ROBUST_INDEX_ALLOC_POOL_RATIO(index_type)
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
                ConfigType.rep_mixed: NUM_REPEATS,
                OptionalConfigType.allocator_pool_ratio: ROBUST_INDEX_ALLOC_POOL_RATIO(index_type)
            }
            for opt_config in EXP_MIX_OPTS:
                configs.append({**common_config, **opt_config})
    # Delete merge space
    for index_type in [IndexType.gpu_masstree, IndexType.gpu_extendhashtable]:
        for prefix_length, key_length in EXP_MERGE_KEY_LENGTHS:
            common_config = {
                ConfigType.index_type: index_type,
                ConfigType.max_keys: DEFAULT_MAXKEY_LONG,
                ConfigType.keylen_prefix: prefix_length,
                ConfigType.keylen_min: key_length,
                ConfigType.keylen_max: key_length,
                ConfigType.num_space: DEFAULT_BATCH_SIZE,
                ConfigType.rep_space: 1,
                OptionalConfigType.allocator_pool_ratio: ROBUST_INDEX_ALLOC_POOL_RATIO(index_type)
            }
            for merge_level in [0, EXP_MAX_MERGE_LEVEL[index_type]]:
                configs.append({**common_config, OptionalConfigType.merge_level: merge_level})
    # Delete merge tput
    for index_type in [IndexType.gpu_masstree, IndexType.gpu_extendhashtable]:
        for merge_level in range(0, EXP_MAX_MERGE_LEVEL[index_type] + 1):
            if index_type == IndexType.gpu_masstree and merge_level in EXP_GPU_MASSTREE_MERGE_SKIP_LEVEL:
                continue # takes tooooooo long
            common_config = {
                ConfigType.index_type: index_type,
                ConfigType.max_keys: DEFAULT_MAXKEY_LONG,
                ConfigType.keylen_prefix: DEFAULT_KEY_LENGTH - 1,
                ConfigType.keylen_min: DEFAULT_KEY_LENGTH,
                ConfigType.keylen_max: DEFAULT_KEY_LENGTH,
                ConfigType.num_insdel: DEFAULT_BATCH_SIZE,
                ConfigType.rep_insdel: NUM_REPEATS,
                OptionalConfigType.allocator_pool_ratio: ROBUST_INDEX_ALLOC_POOL_RATIO(index_type),
                OptionalConfigType.merge_level: merge_level
            }
            configs.append(common_config)
    return configs

if __name__ == "__main__":
    args = parse_args_for_measure()
    configs = generate_configs()
    run_all_and_add_to_json(args, configs, "result_gpu", args.start_from)
