from evaluate import *

NUM_REPEATS = 10
MILLION = 1000000
DEFAULT_MAXKEY_4B = int(4000 * MILLION)
DEFAULT_MAXKEY_LONG = int(400 * MILLION)
DEFAULT_BATCH_SIZE = int(100 * MILLION)
DEFAULT_KEY_LENGHT = 8
DEFAULT_SCAN_COUNT = 10

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
INDEX_TYPES_ORDERED = [
    IndexType.gpu_masstree,
    IndexType.gpu_blink_tree,
    IndexType.cpu_art,
    IndexType.cpu_masstree
]
def INDEX_TYPES_ORDERED_IN(index_types):
    return [i for i in index_types if i in INDEX_TYPES_ORDERED]

GPU_CHAINHT_4BKEYx4B_ALLOC_POOL_RATIO = 0.5
GPU_CHAINHT_LONGKEYx400M_ALLOC_POOL_RATIO = 0.8
GPU_CUCKOOHT_4BKEYx4B_ALLOC_POOL_RATIO = 0.1
GPU_CUCKOOHT_LONGKEYx400M_ALLOC_POOL_RATIO = 0.8

EXP_TABLE_SIZES_4B = [
    int(DEFAULT_MAXKEY_4B / 64),
    int(DEFAULT_MAXKEY_4B / 16),
    int(DEFAULT_MAXKEY_4B / 4),
    int(DEFAULT_MAXKEY_4B),
]
EXP_TABLE_SIZES_LONG = [
    int(DEFAULT_MAXKEY_LONG / 64),
    int(DEFAULT_MAXKEY_LONG / 16),
    int(DEFAULT_MAXKEY_LONG / 4),
    int(DEFAULT_MAXKEY_LONG),
]
EXP_KEY_LENGTHS = [
    1, 2, 4, 8, 16
]
def GPU_MASSTREE_NOSUFFIX_TEST(prefix, keylen):
    if prefix == 0:
        return keylen <= 4
    elif prefix == keylen - 1:
        return True
    assert False
