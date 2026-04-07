from evaluate import *

NUM_REPEATS = 10
MILLION = 1000000
DEFAULT_MAXKEY_4B = int(4000 * MILLION)
DEFAULT_MAXKEY_LONG = int(400 * MILLION)
DEFAULT_BATCH_SIZE = int(100 * MILLION)
DEFAULT_KEY_LENGHT = 8
DEFAULT_SCAN_COUNT = 10
DEFAULT_MIX_READ_RATIO = 0.5

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
IS_INDEX_TYPE_SUPPORT_LONGKEY = INDEX_TYPES_ROBUST + INDEX_TYPES_CPU_BASELINE + [
    IndexType.gpu_dycuckoo
]

GPU_MASSTREE_LONGKEYx400M_ALLOC_POOL_RATIO = 0.8
GPU_CHAINHT_4BKEYx4B_ALLOC_POOL_RATIO = 0.5
GPU_CHAINHT_LONGKEYx400M_ALLOC_POOL_RATIO = 0.8
GPU_CUCKOOHT_4BKEYx4B_ALLOC_POOL_RATIO = 0.1
GPU_CUCKOOHT_LONGKEYx400M_ALLOC_POOL_RATIO = 0.8

EXP_KEY_LENGTHS = [
    1, 2, 4, 8, 16
]
