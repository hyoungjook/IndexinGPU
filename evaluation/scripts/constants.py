from evaluate import *

NUM_REPEATS = 10
MILLION = 1000000
DEFAULT_NUM_KEYS = 10 * MILLION
DEFAULT_KEY_LENGHT = 8
DEFAULT_DELETE_RATIO = 0.1
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

EXP_TABLE_SIZES = [
    int(DEFAULT_NUM_KEYS / 1000),
    int(DEFAULT_NUM_KEYS / 100),
    int(DEFAULT_NUM_KEYS / 10),
    int(DEFAULT_NUM_KEYS)
]
EXP_KEY_LENGTHS = [
    1, 2, 4, 8, 16
]
