from evaluate import *
from constants import *
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

INDEX_LABELS = {
    IndexType.gpu_masstree: "GPUMasstree",
    IndexType.gpu_chainhashtable: "GPUChainHT",
    IndexType.gpu_cuckoohashtable: "GPUCuckooHT",
    IndexType.gpu_extendhashtable: "GPUExtendHT",
    IndexType.gpu_blink_tree: "GPUBtree",
    IndexType.gpu_dycuckoo: "DyCuckoo",
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

def key_length_plots(configs_and_results, plot_file_prefix):
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
    tree_indexes = [i for i in INDEX_TYPES_ROBUST + INDEX_TYPES_GPU_BASELINE if i in IS_INDEX_TYPE_ORDERED]
    hashtable_indexes = [i for i in INDEX_TYPES_ROBUST + INDEX_TYPES_GPU_BASELINE if i not in IS_INDEX_TYPE_ORDERED]
    plot_spec = [
        (0, tree_indexes, ResultType.lookup),
        (1, tree_indexes, ResultType.insert),
        (2, tree_indexes, ResultType.delete),
        (3, tree_indexes, ResultType.mixed),
        (4, tree_indexes, ResultType.scan),
        (5, hashtable_indexes, ResultType.lookup),
        (6, hashtable_indexes, ResultType.insert),
        (7, hashtable_indexes, ResultType.delete),
        (8, hashtable_indexes, ResultType.mixed),
    ]
    for idx, index_types, result_type, in plot_spec:
        fig, ax = plt.subplots(1, 1, figsize=(2, 1.5), constrained_layout=True)
        for index_type in index_types:
            if result_type not in tputs[index_type]:
                continue
            ydata = _convert_mops_to_bops(tputs[index_type][result_type]['avg'])
            if len(ydata) < len(key_lengths_bytes):
                ydata.append(-1)
            xdata = key_lengths_bytes[0:len(ydata)]
            index_label = INDEX_LABELS[index_type]
            line, = ax.plot(
                xdata, ydata,
                label=index_label,
                linewidth=2, markersize=6,
                **INDEX_STYLES[index_type]
            )
            if index_label not in legend_labels:
                legend_handles.append(line)
                legend_labels.append(index_label)
        ax.set_ylim(bottom = 0)
        ax.set_xlim(left = 0)
        _, ymax = ax.get_ylim()
        ytick_candidates = [0.2, 0.5, 1.0]
        for ytick in ytick_candidates:
            num_ticks = int(ymax // ytick)
            if 2 <= num_ticks and num_ticks <= 4:
                yticks = [ytick * x for x in range(num_ticks + 1)]
                break
        ax.set_yticks(yticks)
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))
        ax.set_xticks([0, 20, 40, 60])
        ax.grid(True, which='major', linestyle='--', linewidth=0.6, alpha=0.5)
        plt.savefig(f'{plot_file_prefix}{idx}.pdf', bbox_inches='tight')
        plt.close(fig)
    fig, ax = plt.subplots(1, 1, figsize=(2, 1.5), constrained_layout=True)
    ax.legend(legend_handles, legend_labels, loc='center', ncol = 1)
    ax.axis('off')
    plt.savefig(f'{plot_file_prefix}{len(plot_spec)}.pdf', bbox_inches='tight')
    plt.close(fig)

def generate_plots(args, configs_and_results):
    key_length_plots(configs_and_results, Path(args.result_dir) / 'plot_keylength')

if __name__ == "__main__":
    args = parse_args_for_plot()
    configs_and_results = read_configs_and_results(args)
    generate_plots(args, configs_and_results)
