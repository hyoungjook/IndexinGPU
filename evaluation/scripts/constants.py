from evaluate import *

NUM_REPEATS = 10
MILLION = 1000000
DEFAULT_MAXKEY_LONG = int(500 * MILLION)
DEFAULT_BATCH_SIZE = int(100 * MILLION)
DEFAULT_KEY_LENGTH = 8
DEFAULT_SCAN_COUNT = 10
DEFAULT_MIX_READ_RATIO = 0.5
DEFAULT_SCAN_BATCH_SIZE = int(DEFAULT_BATCH_SIZE // DEFAULT_SCAN_COUNT)

INDEX_TYPES_ROBUST = [
    IndexType.gpu_masstree,
    IndexType.gpu_chainhashtable,
    IndexType.gpu_cuckoohashtable,
    IndexType.gpu_extendhashtable
]
INDEX_TYPES_GPU_BASELINE = [
    IndexType.gpu_blink_tree,
    IndexType.gpu_dycuckoo
]
INDEX_TYPES_CPU_BASELINE = [
    IndexType.cpu_art,
    IndexType.cpu_masstree,
    IndexType.cpu_libcuckoo
]

IS_INDEX_TYPE_ORDERED = [
    IndexType.gpu_masstree,
    IndexType.gpu_blink_tree,
    IndexType.cpu_art,
    IndexType.cpu_masstree
]
IS_INDEX_TYPE_SUPPORT_MIX = INDEX_TYPES_ROBUST + INDEX_TYPES_CPU_BASELINE
def DO_TEST_FOR_INDEX_TYPE(index_type, key_length):
    if index_type == IndexType.gpu_blink_tree and key_length > 1:
        # not support longkeys
        return False
    if index_type == IndexType.gpu_dycuckoo and key_length >= 16:
        # OOM for 500M maxkey
        return False
    return True

def ROBUST_INDEX_ALLOC_POOL_RATIO(index_type):
    if index_type in [IndexType.gpu_masstree, IndexType.gpu_extendhashtable]:
        return 0.9
    elif index_type in [IndexType.gpu_chainhashtable]:
        return 0.85
    elif index_type in [IndexType.gpu_cuckoohashtable]:
        return 0.8
    assert False

EXP_KEY_LENGTHS = [
    1, 2, 4, 8, 16
]

EXP_GPU_MASSTREE_OPTS = [
    {
        ConfigType.keylen_prefix: DEFAULT_KEY_LENGTH - 1,
        OptionalConfigType.enable_suffix: 0,
        OptionalConfigType.lookup_concurrent: 0,
        OptionalConfigType.reuse_root: 0
    },
    #{  ##### This config gives OOM, so instead None #####
    #    ConfigType.keylen_prefix: 0,
    #    OptionalConfigType.enable_suffix: 0,
    #    OptionalConfigType.lookup_concurrent: 0,
    #    OptionalConfigType.reuse_root: 0
    #},
    None,
    {
        ConfigType.keylen_prefix: 0,
        OptionalConfigType.enable_suffix: 1,
        OptionalConfigType.lookup_concurrent: 0,
        OptionalConfigType.reuse_root: 0
    },
]
EXP_GPU_EXTENDHT_OPTS = [
    {
        ConfigType.keylen_prefix: 0,
        OptionalConfigType.hash_tag_level: 0,
        OptionalConfigType.lookup_concurrent: 0,
        OptionalConfigType.reuse_dirsize: 0
    },
    {
        ConfigType.keylen_prefix: DEFAULT_KEY_LENGTH - 1,
        OptionalConfigType.hash_tag_level: 0,
        OptionalConfigType.lookup_concurrent: 0,
        OptionalConfigType.reuse_dirsize: 0
    },
    {
        ConfigType.keylen_prefix: DEFAULT_KEY_LENGTH - 1,
        OptionalConfigType.hash_tag_level: 1,
        OptionalConfigType.lookup_concurrent: 0,
        OptionalConfigType.reuse_dirsize: 0
    },
    {
        ConfigType.keylen_prefix: DEFAULT_KEY_LENGTH - 1,
        OptionalConfigType.hash_tag_level: 2,
        OptionalConfigType.lookup_concurrent: 0,
        OptionalConfigType.reuse_dirsize: 0
    },
]

EXP_MIX_READ_RATIOS = [
    0, 0.25, 0.5, 0.75, 1
]
EXP_MIX_OPTS = [
    {
        OptionalConfigType.tile_size: 32,
        ConfigType.mix_presort: 0,
    },
    {
        OptionalConfigType.tile_size: 16,
        ConfigType.mix_presort: 0,
    },
    {
        OptionalConfigType.tile_size: 16,
        ConfigType.mix_presort: 1,
    },
]

EXP_MERGE_LEVELS = {
    IndexType.gpu_masstree: [0, 3],
    IndexType.gpu_extendhashtable: [0, 2],
}
EXP_MERGE_KEY_LENGTHS = [
    1, DEFAULT_KEY_LENGTH
]
EXP_MERGE_ERASE_RATIOS = [
    float(x / DEFAULT_MAXKEY_LONG) for x in range(0, DEFAULT_MAXKEY_LONG + 1, DEFAULT_BATCH_SIZE)
]
