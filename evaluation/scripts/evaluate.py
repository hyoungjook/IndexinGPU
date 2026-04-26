import argparse
from enum import Enum, auto
import glob
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile

class BenchExecutable(Enum):
    robust = auto()
    gpu_baseline = auto()
    cpu_baseline = auto()

class IndexType(Enum):
    gpu_masstree = auto()
    gpu_chainhashtable = auto()
    gpu_cuckoohashtable = auto()
    gpu_extendhashtable = auto()
    gpu_blink_tree = auto()
    gpu_dycuckoo = auto()
    cpu_libcuckoo = auto()
    cpu_masstree = auto()
    cpu_art = auto()

class ConfigType(Enum):
    max_keys = auto()
    keylen_prefix = auto()
    keylen_min = auto()
    keylen_max = auto()
    keylen_theta = auto()
    num_lookups = auto()
    lookup_theta = auto()
    num_scans = auto()
    scan_count = auto()
    num_insdel = auto()
    num_mixed = auto()
    num_space = auto()
    mix_read_ratio = auto()
    mix_presort = auto()
    rep_lookup = auto()
    rep_scan = auto()
    rep_insdel = auto()
    rep_mixed = auto()
    rep_space = auto()
    index_type = auto()
    only_check_space = auto()
    use_pinned_host_memory = auto()

class OptionalConfigType(Enum):
    allocator_pool_ratio = auto()
    tile_size = auto()
    lookup_concurrent = auto()
    enable_suffix = auto()
    merge_level = auto()
    initial_array_fill_factor = auto()
    use_hash_tag = auto()
    merge_chains = auto()
    initial_directory_size = auto()
    resize_policy = auto()
    load_factor_threshold = auto()
    hash_tag_level = auto()
    erase_concurrent = auto()
    use_lock = auto()
    initial_capacity = auto()
    use_shmem_key = auto()

class ResultType(Enum):
    insert = auto()
    delete = auto()
    lookup = auto()
    scan = auto()
    mixed = auto()
    space = auto()

EXECUTABLE_INFO = {
    BenchExecutable.robust: {
        'path': 'bin/universal_bench',
        'indexes': [
            IndexType.gpu_masstree,
            IndexType.gpu_chainhashtable,
            IndexType.gpu_cuckoohashtable,
            IndexType.gpu_extendhashtable,
        ]
    },
    BenchExecutable.gpu_baseline: {
        'path': 'bin/universal_bench_with_gpu_baseline',
        'indexes': [
            IndexType.gpu_blink_tree,
            IndexType.gpu_dycuckoo,
        ]
    },
    BenchExecutable.cpu_baseline: {
        'path': 'bin/universal_bench_with_cpu_baseline',
        'indexes': [
            IndexType.cpu_libcuckoo,
            IndexType.cpu_masstree,
            IndexType.cpu_art,
        ]
    }
}

INDEX_INFO = {
    IndexType.gpu_masstree: [
        OptionalConfigType.allocator_pool_ratio,
        OptionalConfigType.tile_size,
        OptionalConfigType.lookup_concurrent,
        OptionalConfigType.enable_suffix,
        OptionalConfigType.merge_level,
        OptionalConfigType.use_shmem_key,
    ],
    IndexType.gpu_chainhashtable: [
        OptionalConfigType.allocator_pool_ratio,
        OptionalConfigType.tile_size,
        OptionalConfigType.lookup_concurrent,
        OptionalConfigType.initial_array_fill_factor,
        OptionalConfigType.use_hash_tag,
        OptionalConfigType.merge_chains,
        OptionalConfigType.use_shmem_key,
    ],
    IndexType.gpu_cuckoohashtable: [
        OptionalConfigType.allocator_pool_ratio,
        OptionalConfigType.tile_size,
        OptionalConfigType.lookup_concurrent,
        OptionalConfigType.initial_array_fill_factor,
        OptionalConfigType.use_hash_tag,
        OptionalConfigType.use_shmem_key,
    ],
    IndexType.gpu_extendhashtable: [
        OptionalConfigType.allocator_pool_ratio,
        OptionalConfigType.tile_size,
        OptionalConfigType.lookup_concurrent,
        OptionalConfigType.initial_directory_size,
        OptionalConfigType.resize_policy,
        OptionalConfigType.load_factor_threshold,
        OptionalConfigType.hash_tag_level,
        OptionalConfigType.merge_level,
        OptionalConfigType.use_shmem_key,
    ],
    IndexType.gpu_blink_tree: [
        OptionalConfigType.lookup_concurrent,
        OptionalConfigType.erase_concurrent
    ],
    IndexType.gpu_dycuckoo: [
        OptionalConfigType.use_lock,
        OptionalConfigType.initial_capacity
    ],
    IndexType.cpu_libcuckoo: [
        OptionalConfigType.initial_capacity
    ],
    IndexType.cpu_masstree: [],
    IndexType.cpu_art: []
}

def get_executable_of_index(index_type: IndexType) -> BenchExecutable:
    for executable_type, info in EXECUTABLE_INFO.items():
        if index_type in info['indexes']:
            return executable_type
    assert False

def config_value_to_str(value) -> str:
    if isinstance(value, Enum):
        return value.name
    return str(value)

def parse_args_for_measure():
    parser = argparse.ArgumentParser()
    parser.add_argument('--build-dir', type=str,
        default=str(Path(__file__).parent.parent.parent / 'build'),
        help='Path of the program build directory.')
    parser.add_argument('--result-dir', type=str, required=True,
        help='Path of directory to add the result JSON file.')
    parser.add_argument('--start-from', type=int, default=0,
        help='Skip n configs at the beginning, used for debugging')
    args = parser.parse_args()
    return args

def run_one(args, config):
    # parse config
    index_type = config[ConfigType.index_type]
    executable_type = get_executable_of_index(index_type)
    executable_path = EXECUTABLE_INFO[executable_type]['path']
    executable_path = str(Path(args.build_dir) / executable_path)
    cmd = []
    cmd.append(str(executable_path))
    for config_type in ConfigType:
        if config_type in config:
            config_cmd = config_type.name
            config_value = config_value_to_str(config[config_type])
            cmd.append(f'{config_cmd}={config_value}')
    for optional_config_type in INDEX_INFO[index_type]:
        if optional_config_type in config:
            config_cmd = optional_config_type.name
            config_value = config_value_to_str(config[optional_config_type])
            cmd.append(f'{config_cmd}={config_value}')
    cmd.append('print_all_measurements=1')
    # execute subprocess
    print(' '.join(cmd))
    with tempfile.NamedTemporaryFile(mode='w', delete=False) as log:
        log_file = log.name
        print(f'Writing execution log to {log_file}')
        log.write(f'CMD: {cmd}\n\n')
        log.flush()
        ret = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT)
        ret.check_returncode()
    # parse result
    with open(log_file, 'r') as f:
        result_str = f.read()
    result_lines = result_str.split('\n')
    result = {}
    parsed_config = {}
    space_results = []
    for result_line in result_lines:
        if 'CMD' in result_line:
            pass
        elif '=' in result_line:
            result_tokens = result_line.strip().split('=')
            assert len(result_tokens) == 2
            config_name = result_tokens[0]
            config_value = result_tokens[1]
            parsed_config[config_name] = config_value
        elif 'Mop/s' in result_line:
            result_tokens = result_line.split(' ')
            assert len(result_tokens) >= 5
            result_type = result_tokens[0][:-1]
            assert result_type in [r.name for r in ResultType]
            avg_value = float(result_tokens[1])
            raw_values = []
            num_values = int(re.sub(r'[();]', '', result_tokens[3]))
            for i in range(num_values):
                raw_values.append(float(result_tokens[4 + i]))
            result[result_type] = {'avg': avg_value,
                                   'raw': raw_values}
        elif 'simple_slab' in result_line:
            result_tokens = result_line.split(' ')
            for token in result_tokens:
                if '%' in token:
                    space_results.append(float(re.sub(r'[()%]', '', token)))
                    break
    if len(space_results) > 0:
        result['space'] = space_results
    print(f'parsed_config: {parsed_config}, result: {result}')
    return parsed_config, result

def run_all_and_add_to_json(args, configs, result_file_name, start_from):
    # run all
    os.makedirs(args.result_dir, exist_ok=True)
    result_file = Path(args.result_dir) / f'{result_file_name}.json'
    for idx in range(start_from, len(configs)):
        print(f'Experiment {idx} (total {len(configs)})')
        config = configs[idx]
        parsed_config, result = run_one(args, config)
        for config_type, config_value in config.items():
            assert config_type.name in parsed_config and \
                parsed_config[config_type.name] == config_value_to_str(config_value)
        # add to json
        json_data = []
        if result_file.exists():
            try:
                with result_file.open("r", encoding="utf-8") as f:
                    json_data = json.load(f)
                if not isinstance(json_data, list):
                    raise ValueError("JSON root must be a list")
            except (json.JSONDecodeError, OSError):
                json_data = []
        json_data.append({'config': parsed_config, 'result': result})
        with result_file.open("w", encoding="utf-8") as f:
            json.dump(json_data, f, ensure_ascii=False, indent=2)

def parse_args_for_plot():
    parser = argparse.ArgumentParser()
    parser.add_argument('--result-dir', type=str, required=True,
        help='Path of directory with JSON files with the result.')
    args = parser.parse_args()
    return args

def read_configs_and_results(args):
    json_data = []
    for result_file in glob.glob(os.path.join(args.result_dir, '*.json')):
        try:
            with Path(result_file).open("r", encoding="utf-8") as f:
                data = json.load(f)
                assert isinstance(data, list)
                json_data += data
        except Exception as e:
            pass
    if len(json_data) == 0:
        print(f'No valid json result file in {args.result_dir}')
        exit(1)
    return json_data

def filter(configs_and_results: list[dict], desired_config: dict, result_type: ResultType):
    for config_and_result in configs_and_results:
        config = config_and_result['config']
        result = config_and_result['result']
        match = True
        for desired_config_type, desired_config_value in desired_config.items():
            if config.get(desired_config_type.name) != config_value_to_str(desired_config_value):
                match = False
                break
        if not match:
            continue
        if result_type.name not in result:
            continue
        return result
    print(f'filter fail: {desired_config} {result_type}')
    assert False
