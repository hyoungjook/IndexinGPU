from evaluate import *
from constants import *
import matplotlib.patches as mpatch
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

INDEX_LABELS = {
    IndexType.gpu_masstree: "GPUMasstree",
    IndexType.gpu_chainhashtable: "GPUChainHT",
    IndexType.gpu_cuckoohashtable: "GPUCuckooHT",
    IndexType.gpu_extendhashtable: "GPUExtendHT",
    IndexType.gpu_blink_tree: "GPUBtree",
    IndexType.gpu_dycuckoo: "GPUDyCuckoo",
    IndexType.cpu_art: "(CPU)ART",
    IndexType.cpu_masstree: "(CPU)Masstree",
    IndexType.cpu_libcuckoo: "(CPU)Libcuckoo",
}
INDEX_STYLES = {
    IndexType.gpu_masstree: {"color": "#0B6E4F", "marker": "o", "linestyle": "-"},
    IndexType.gpu_chainhashtable: {"color": "#D1495B", "marker": "s", "linestyle": "-"},
    IndexType.gpu_cuckoohashtable: {"color": "#00798C", "marker": "^", "linestyle": "-"},
    IndexType.gpu_extendhashtable: {"color": "#EDAE49", "marker": "D", "linestyle": "-"},
    IndexType.gpu_blink_tree: {"color": "#8F2D56", "marker": "<", "linestyle": ":"},
    IndexType.gpu_dycuckoo: {"color": "#3B60E4", "marker": "h", "linestyle": ":"},
    IndexType.cpu_art: {"color": "#5C4D7D", "marker": "P", "linestyle": ":"},
    IndexType.cpu_masstree: {"color": "#6C9A8B", "marker": "X", "linestyle": ":"},
    IndexType.cpu_libcuckoo: {"color": "#9C6644", "marker": "v", "linestyle": ":"},
    #IndexType_gpu_masstree_no_suffix: {"color": "#549A84", "marker": "*", "linestyle": "-"},
    #IndexType_gpu_extendht_no_hashtag: {"color": "#CE973E", "marker": "d", "linestyle": "-"},
    #IndexType_gpu_dycuckoo_with_lock: {"color": "#7993EF", "marker": "H", "linestyle": ":"},
}

def _table_size_label(value, _):
    if value >= MILLION:
        return f"{int(value / MILLION)}M"
    if value >= 1000:
        return f"{int(value / 1000)}K"
    return str(int(value))

def _convert_mops_to_bops(values):
        return [v / 1000 for v in values]

def key_length_plots(configs_and_results, plot_file):
    tputs = {}
    for index_type in INDEX_TYPES_ROBUST + INDEX_TYPES_GPU_BASELINE:
        tputs[index_type] = {}
        result_types = [ResultType.lookup, ResultType.insert, ResultType.delete]
        if index_type in IS_INDEX_TYPE_SUPPORT_MIX:
            result_types.append(ResultType.mixed)
        if index_type in IS_INDEX_TYPE_ORDERED:
            result_types.append(ResultType.scan)
        if index_type in IS_INDEX_TYPE_SUPPORT_LONGKEY:
            key_lengths = EXP_KEY_LENGTHS
        else:
            key_lengths = [1]
        for result_type in result_types:
            tputs[index_type][result_type] = {
                'avg': [], 'min': [], 'max': []
            }
            for key_length in key_lengths:
                desired_config = {
                    ConfigType.index_type: index_type,
                    ConfigType.max_keys: DEFAULT_MAXKEY_LONG,
                    ConfigType.keylen_prefix: 0,
                    ConfigType.keylen_min: key_length,
                    ConfigType.keylen_max: key_length,
                }
                if result_type == ResultType.lookup:
                    desired_config[ConfigType.num_lookups] = DEFAULT_BATCH_SIZE
                elif result_type in [ResultType.insert, ResultType.delete]:
                    desired_config[ConfigType.num_insdel] = DEFAULT_BATCH_SIZE
                elif result_type == ResultType.mixed:
                    desired_config[ConfigType.num_mixed] = DEFAULT_BATCH_SIZE
                    desired_config[ConfigType.mix_read_ratio] = DEFAULT_MIX_READ_RATIO
                elif result_type == ResultType.scan:
                    desired_config[ConfigType.num_scans] = DEFAULT_BATCH_SIZE
                    desired_config[ConfigType.scan_count] = DEFAULT_SCAN_COUNT
                result = filter(configs_and_results, desired_config, result_type)
                for metric_type in ['avg', 'min', 'max']:
                    tputs[index_type][result_type][metric_type].append(float(result[result_type.name][metric_type]))
    # plot trees
    key_lengths_bytes = [4 * l for l in EXP_KEY_LENGTHS]
    legend_handles = []
    legend_labels = []
    fig, axes = plt.subplots(2, 5, figsize=(12, 4), constrained_layout=True)
    index_types = [i for i in INDEX_TYPES_ROBUST + INDEX_TYPES_GPU_BASELINE if i in IS_INDEX_TYPE_ORDERED]
    result_types = [ResultType.lookup, ResultType.insert, ResultType.delete, ResultType.mixed, ResultType.scan]
    for i in range(len(result_types)):
        result_type = result_types[i]
        for index_type in index_types:
            if result_type not in tputs[index_type]:
                continue
            ydata = _convert_mops_to_bops(tputs[index_type][result_type]['avg'])
            if len(ydata) < len(key_lengths_bytes):
                ydata.append(-1)
            xdata = key_lengths_bytes[0:len(ydata)]
            index_label = INDEX_LABELS[index_type]
            line, = axes[0, i].plot(
                xdata, ydata,
                label=index_label,
                linewidth=2, markersize=6,
                **INDEX_STYLES[index_type]
            )
            if index_label not in legend_labels:
                legend_handles.append(line)
                legend_labels.append(index_label)
        axes[0, i].set_ylim(bottom = 0)
        axes[0, i].set_title(result_type.name)
        axes[0, i].grid(True, which='major', linestyle='--', linewidth=0.6, alpha=0.5)
    axes[0, 0].set_ylabel('Throughput (Bop/s)')
    index_types = [i for i in INDEX_TYPES_ROBUST + INDEX_TYPES_GPU_BASELINE if i not in IS_INDEX_TYPE_ORDERED]
    result_types = [ResultType.lookup, ResultType.insert, ResultType.delete, ResultType.mixed]
    for i in range(len(result_types)):
        result_type = result_types[i]
        for index_type in index_types:
            if result_type not in tputs[index_type]:
                continue
            ydata = _convert_mops_to_bops(tputs[index_type][result_type]['avg'])
            xdata = key_lengths_bytes[0:len(ydata)]
            index_label = INDEX_LABELS[index_type]
            line, = axes[1, i].plot(
                xdata, ydata,
                label=index_label,
                linewidth=2, markersize=6,
                **INDEX_STYLES[index_type]
            )
            if index_label not in legend_labels:
                legend_handles.append(line)
                legend_labels.append(index_label)
        axes[1, i].set_ylim(bottom = 0)
        axes[1, i].set_xlabel('Key Length (B)')
        axes[1, i].grid(True, which='major', linestyle='--', linewidth=0.6, alpha=0.5)
            
    axes[1, 0].set_ylabel('Throughput (Bop/s)')
    fig.legend(legend_handles, legend_labels, loc='lower center', ncol=7, bbox_to_anchor=(0.5, 1))
    plt.savefig(plot_file, bbox_inches='tight')
    plt.close(fig)

def generate_plots(args, configs_and_results):
    key_length_plots(configs_and_results, Path(args.result_dir) / 'plot_keylengths.pdf')

if __name__ == "__main__":
    args = parse_args_for_plot()
    configs_and_results = read_configs_and_results(args)
    generate_plots(args, configs_and_results)
