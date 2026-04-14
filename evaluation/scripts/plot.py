from evaluate import *
from constants import *
import matplotlib.legend_handler as mlegh
import matplotlib.lines as mline
import matplotlib.patches as mpatch
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

GPU_VM_HOURLY_PRICE = 3.673
CPU_VM_HOURLY_PRICE = 3.648
CPU_BASELINE_ADJUST = GPU_VM_HOURLY_PRICE / CPU_VM_HOURLY_PRICE

INDEX_LABELS = {
    IndexType.gpu_masstree: "GPUMasstree",
    IndexType.gpu_chainhashtable: "GPUChainHT",
    IndexType.gpu_cuckoohashtable: "GPUCuckooHT",
    IndexType.gpu_extendhashtable: "GPUExtendHT",
    IndexType.gpu_blink_tree: "GPUBtree",
    IndexType.gpu_dycuckoo: "DyCuckoo",
    IndexType.cpu_art: "ART",
    IndexType.cpu_masstree: "Masstree",
    IndexType.cpu_libcuckoo: "Libcuckoo",
}
INDEX_STYLES = {
    IndexType.gpu_masstree: {"color": "#0B6E4F", "marker": "o", "linestyle": "-"},
    IndexType.gpu_chainhashtable: {"color": "#D1495B", "marker": "s", "linestyle": "-"},
    IndexType.gpu_cuckoohashtable: {"color": "#00798C", "marker": "^", "linestyle": "-"},
    IndexType.gpu_extendhashtable: {"color": "#EDAE49", "marker": "D", "linestyle": "-"},
    IndexType.gpu_blink_tree: {"color": "#8F2D56", "marker": "<", "linestyle": ":"},
    IndexType.gpu_dycuckoo: {"color": "#3B60E4", "marker": "h", "linestyle": ":"},
    IndexType.cpu_art: {"color": "#5C4D7D", "marker": "P", "linestyle": "--"},
    IndexType.cpu_masstree: {"color": "#6C9A8B", "marker": "X", "linestyle": "--"},
    IndexType.cpu_libcuckoo: {"color": "#9C6644", "marker": "v", "linestyle": "--"},
}
HATCH_STYLES = {
    IndexType.gpu_masstree: 'o',
    IndexType.gpu_extendhashtable: 'x',
    IndexType.cpu_art: '///',
    IndexType.cpu_masstree: '\\\\\\',
    IndexType.cpu_libcuckoo: '|||',
    IndexType.gpu_dycuckoo: '---',
}

def _convert_mops_to_bops(values, index_type):
    bops = [None if v is None else v / 1000 for v in values]
    if index_type in INDEX_TYPES_CPU_BASELINE:
        bops = [v * CPU_BASELINE_ADJUST for v in bops]
    return bops

def _add_throughput_error_bars(ax, xdata, avg_values, min_values, max_values, *, color, for_barplot=False):
    # Some throughput plots include placeholders (for example OOM entries), so skip
    # error bars when parsed min/max values are unavailable for those points.
    if min_values is None or max_values is None:
        return
    error_xdata = []
    error_ydata = []
    lower_errors = []
    upper_errors = []
    for xvalue, avg_value, min_value, max_value in zip(xdata, avg_values, min_values, max_values):
        if avg_value is None or min_value is None or max_value is None:
            continue
        error_xdata.append(xvalue)
        error_ydata.append(avg_value)
        lower_errors.append(max(avg_value - min_value, 0))
        upper_errors.append(max(max_value - avg_value, 0))
    if not error_xdata:
        return
    ax.errorbar(
        error_xdata, error_ydata,
        yerr=[lower_errors, upper_errors],
        fmt='none',
        ecolor=color,
        elinewidth=1,
        capsize=5 if for_barplot else 3,
        capthick=1,
        zorder=4,
    )

def _make_fixed_plot_area_figure(plot_width, plot_height, *, include_xlabel=False, include_ylabel=False):
    # Keep the drawable axes area fixed and grow the outer figure only when labels are present.
    left_margin = 0.56 if include_ylabel else 0.32
    bottom_margin = 0.42 if include_xlabel else 0.25
    right_margin = 0.08
    top_margin = 0.06
    fig_width = left_margin + plot_width + right_margin
    fig_height = bottom_margin + plot_height + top_margin
    fig = plt.figure(figsize=(fig_width, fig_height))
    ax = fig.add_axes([
        left_margin / fig_width,
        bottom_margin / fig_height,
        plot_width / fig_width,
        plot_height / fig_height,
    ])
    return fig, ax

def key_length_plots(configs_and_results, plot_file_prefix):
    tputs = {}
    all_index_types = INDEX_TYPES_ROBUST + INDEX_TYPES_GPU_BASELINE + INDEX_TYPES_CPU_BASELINE
    for index_type in all_index_types:
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
                    desired_config[ConfigType.num_scans] = DEFAULT_SCAN_BATCH_SIZE
                    desired_config[ConfigType.scan_count] = DEFAULT_SCAN_COUNT
                result = filter(configs_and_results, desired_config, result_type)
                for metric_type in ['avg', 'min', 'max']:
                    tputs[index_type][result_type][metric_type].append(float(result[result_type.name][metric_type]))
    # plot trees
    key_lengths_bytes = [4 * l for l in EXP_KEY_LENGTHS]
    legend_handles = []
    legend_labels = []
    tree_indexes = [i for i in all_index_types if i in IS_INDEX_TYPE_ORDERED]
    hashtable_indexes = [i for i in all_index_types if i not in IS_INDEX_TYPE_ORDERED]
    plot_spec = [
        (0, tree_indexes, ResultType.lookup, False, True),
        (1, tree_indexes, ResultType.insert, False, False),
        (2, tree_indexes, ResultType.delete, False, False),
        (3, tree_indexes, ResultType.mixed, False, False),
        (4, tree_indexes, ResultType.scan, False, False),
        (5, hashtable_indexes, ResultType.lookup, True, True),
        (6, hashtable_indexes, ResultType.insert, True, False),
        (7, hashtable_indexes, ResultType.delete, True, False),
        (8, hashtable_indexes, ResultType.mixed, True, False),
    ]
    plot_names = [
        'tree-lookup', 'tree-insert', 'tree-delete', 'tree-mixed', 'tree-scan',
        'ht-lookup', 'ht-insert', 'ht-delete', 'ht-mixed'
    ]
    for idx, index_types, result_type, set_xlabel, set_ylabel in plot_spec:
        fig, ax = _make_fixed_plot_area_figure(2, 1.3,
            include_xlabel=set_xlabel,
            include_ylabel=set_ylabel,
        )
        for index_type in index_types:
            if result_type not in tputs[index_type]:
                continue
            avg_values = _convert_mops_to_bops(tputs[index_type][result_type]['avg'], index_type)
            min_values = _convert_mops_to_bops(tputs[index_type][result_type]['min'], index_type)
            max_values = _convert_mops_to_bops(tputs[index_type][result_type]['max'], index_type)
            ydata = avg_values.copy()
            markevery = range(len(ydata))
            if len(markevery) == 1:
                ydata.append(ydata[0] * 0.7)
            xdata = key_lengths_bytes[0:len(ydata)]
            index_label = INDEX_LABELS[index_type]
            line, = ax.plot(
                xdata, ydata,
                label=index_label,
                markevery=markevery,
                linewidth=2, markersize=6,
                **INDEX_STYLES[index_type]
            )
            _add_throughput_error_bars(
                ax,
                xdata[0:len(avg_values)],
                avg_values,
                min_values,
                max_values,
                color=INDEX_STYLES[index_type]['color']
            )
            if len(markevery) == 1:
                ax.text(xdata[1], ydata[1], "X", fontsize=10, color='red', fontweight='bold', ha='center', va='center')
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
        if set_xlabel:
            ax.set_xlabel('Key Length (B)')
        if set_ylabel:
            ax.set_ylabel(r'Throughput ($10^9$/s)')
        fig.savefig(f'{plot_file_prefix}-{plot_names[idx]}.pdf', bbox_inches='tight')
        plt.close(fig)
    fig, ax = plt.subplots(1, 1, figsize=(2.3, 2), constrained_layout=True)
    ax.legend(legend_handles, legend_labels, loc='center', ncol=1, handlelength=3)
    ax.axis('off')
    plt.savefig(f'{plot_file_prefix}-legend.pdf', bbox_inches='tight')
    plt.close(fig)

def suffix_plots(configs_and_results, plot_file_prefix):
    tputs = {}
    plot_specs = [
        (0, IndexType.gpu_masstree, ResultType.lookup, EXP_GPU_MASSTREE_OPTS, DEFAULT_MAXKEY_LONG),
        (1, IndexType.gpu_extendhashtable, ResultType.lookup, EXP_GPU_EXTENDHT_OPTS, 2 * DEFAULT_BATCH_SIZE),
        (2, IndexType.gpu_extendhashtable, ResultType.insert, EXP_GPU_EXTENDHT_OPTS, 2 * DEFAULT_BATCH_SIZE),
    ]
    plot_names = [
        'mt-lookup', 'et-lookup', 'et-insert'
    ]
    for idx, index_type, result_type, opt_configs, max_key in plot_specs:
        tputs[idx] = {
            'avg': [], 'min': [], 'max': []
        }
        desired_config = {
            ConfigType.index_type: index_type,
            ConfigType.max_keys: max_key,
            ConfigType.keylen_min: DEFAULT_KEY_LENGTH,
            ConfigType.keylen_max: DEFAULT_KEY_LENGTH,
        }
        if result_type == ResultType.lookup:
            desired_config[ConfigType.num_lookups] = DEFAULT_BATCH_SIZE
        else:
            desired_config[ConfigType.num_insdel] = DEFAULT_BATCH_SIZE
        for opt_config in opt_configs:
            if opt_config is None:
                tputs[idx]['avg'].append(0)
                tputs[idx]['min'].append(None)
                tputs[idx]['max'].append(None)
            else:
                result = filter(configs_and_results, {**desired_config, **opt_config}, result_type)
                for metric_type in ['avg', 'min', 'max']:
                    tputs[idx][metric_type].append(float(result[result_type.name][metric_type]))
    # plot
    mt_xticklabels = ['C', 'R', 'RS']
    et_xticklabels = ['R', 'C', 'CT', 'Ct']
    for idx, index_type, result_type, opt_configs, _ in plot_specs:
        fig, ax = _make_fixed_plot_area_figure(1.3, 1.1,
            include_xlabel=False, include_ylabel=(idx == 0))
        avg_values = _convert_mops_to_bops(tputs[idx]['avg'], index_type)
        min_values = _convert_mops_to_bops(tputs[idx]['min'], index_type)
        max_values = _convert_mops_to_bops(tputs[idx]['max'], index_type)
        ydata = avg_values
        xlabel = mt_xticklabels if index_type == IndexType.gpu_masstree else et_xticklabels
        xdata = range(len(xlabel))
        ax.bar(xdata, ydata,
            fill=False,
            edgecolor=INDEX_STYLES[index_type]['color'],
            hatch=HATCH_STYLES[index_type],
            linewidth=2
        )
        _add_throughput_error_bars(
            ax,
            xdata,
            avg_values,
            min_values,
            max_values,
            color=INDEX_STYLES[index_type]['color'],
            for_barplot=True
        )
        if idx == 0:
            ax.text(xdata[1], 0, "OOM", fontsize=10, color='red', fontweight='bold', ha='center', va='bottom')
        ax.set_ylim(bottom=0)
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))
        ax.set_xticks(xdata)
        ax.set_xticklabels(xlabel)
        ax.grid(True, axis='y', which='major', linestyle='--', linewidth=0.6, alpha=0.5)
        if idx == 0:
            ax.set_ylabel(r'Throughput ($10^9$/s)')
        plt.savefig(f'{plot_file_prefix}-{plot_names[idx]}.pdf', bbox_inches='tight')
        plt.close(fig)

def tile_plots(configs_and_results, plot_file_prefix):
    tputs = {}
    index_types = [IndexType.gpu_masstree, IndexType.gpu_extendhashtable]
    plot_names = ['mt', 'et']
    for index_type in index_types:
        for opt_idx, opt_config in enumerate(EXP_MIX_OPTS):
            tputs[(index_type, opt_idx)] = {
                'avg': [], 'min': [], 'max': []
            }
            for mix_read_ratio in EXP_MIX_READ_RATIOS:
                desired_config = {
                    ConfigType.index_type: index_type,
                    ConfigType.max_keys: DEFAULT_MAXKEY_LONG,
                    ConfigType.keylen_prefix: 0,
                    ConfigType.keylen_min: DEFAULT_KEY_LENGTH,
                    ConfigType.keylen_max: DEFAULT_KEY_LENGTH,
                    ConfigType.num_mixed: DEFAULT_BATCH_SIZE,
                    ConfigType.mix_read_ratio: mix_read_ratio,
                }
                result = filter(configs_and_results, {**desired_config, **opt_config}, ResultType.mixed)
                for metric_type in ['avg', 'min', 'max']:
                    tputs[(index_type, opt_idx)][metric_type].append(float(result['mixed'][metric_type]))
    # plot
    labels = ['FullWarp', 'HalfWarp', 'HalfWarp+PreSort']
    styles = [
        {"color": "#549A84", "marker": "v", "linestyle": "-"},
        {"color": "#CE973E", "marker": "d", "linestyle": "-"},
        {"color": "#7993EF", "marker": "H", "linestyle": "-"},
    ]
    legend_handles = []
    legend_labels = []
    for idx, index_type in enumerate(index_types):
        fig, ax = _make_fixed_plot_area_figure(2, 1.3,
            include_xlabel=True,
            include_ylabel=(idx == 0)
        )
        for opt_idx in range(len(EXP_MIX_OPTS)):
            avg_values = _convert_mops_to_bops(tputs[(index_type, opt_idx)]['avg'], index_type)
            min_values = _convert_mops_to_bops(tputs[(index_type, opt_idx)]['min'], index_type)
            max_values = _convert_mops_to_bops(tputs[(index_type, opt_idx)]['max'], index_type)
            ydata = avg_values
            xdata = EXP_MIX_READ_RATIOS
            line, = ax.plot(
                xdata, ydata,
                label=labels[opt_idx],
                linewidth=2, markersize=6,
                **styles[opt_idx]
            )
            _add_throughput_error_bars(
                ax,
                xdata,
                avg_values,
                min_values,
                max_values,
                color=styles[opt_idx]['color'],
            )
            if labels[opt_idx] not in legend_labels:
                legend_handles.append(line)
                legend_labels.append(labels[opt_idx])
        ax.set_ylim(bottom = 0)
        ax.set_xlim(left=0, right=1)
        _, ymax = ax.get_ylim()
        ytick_candidates = [0.2, 0.4]
        for ytick in ytick_candidates:
            num_ticks = int(ymax // ytick)
            if 2 <= num_ticks and num_ticks <= 3:
                yticks = [ytick * x for x in range(num_ticks + 1)]
                break
        ax.set_yticks(yticks)
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))
        ax.set_xlabel('Lookup Ratio')
        if idx == 0:
            ax.set_ylabel(r'Throughput ($10^9$/s)')
        ax.grid(True, which='major', linestyle='--', linewidth=0.6, alpha=0.5)
        plt.savefig(f'{plot_file_prefix}-{plot_names[idx]}.pdf', bbox_inches='tight')
        plt.close(fig)
    fig, ax = plt.subplots(1, 1, figsize=(4, 0.3), constrained_layout=True)
    ax.legend(legend_handles, legend_labels, loc='center', ncol=len(legend_labels))
    ax.axis('off')
    plt.savefig(f'{plot_file_prefix}-legend.pdf', bbox_inches='tight')
    plt.close(fig)

def merge_plots(configs_and_results, plot_file_prefix):
    tputs = {}
    spaces = {}
    index_types = [IndexType.gpu_masstree, IndexType.gpu_extendhashtable]
    plot_names = ['mt', 'et']
    for index_type in index_types:
        for key_length in EXP_MERGE_KEY_LENGTHS:
            for merge_level in EXP_MERGE_LEVELS[index_type]:
                tputs[(index_type, key_length, merge_level)] = {
                    'avg': [], 'min': [], 'max': []
                }
                spaces[(index_type, key_length, merge_level)] = []
                desired_config = {
                    ConfigType.index_type: index_type,
                    ConfigType.max_keys: DEFAULT_MAXKEY_LONG,
                    ConfigType.keylen_prefix: 0,
                    ConfigType.keylen_min: key_length,
                    ConfigType.keylen_max: key_length,
                    ConfigType.num_space: DEFAULT_BATCH_SIZE,
                    OptionalConfigType.merge_level: merge_level,
                }
                result = filter(configs_and_results, desired_config, ResultType.delete_space)
                tputs[(index_type, key_length, merge_level)] = {
                    'avg': [float(v) for v in result['delete_space']['avg']],
                    'min': [float(v) for v in result['delete_space']['min']],
                    'max': [float(v) for v in result['delete_space']['max']]
                }
                result = filter(configs_and_results, desired_config, ResultType.space)
                spaces[(index_type, key_length, merge_level)] = [float(v) for v in result['space']]
    # plot
    labels = {
        (0, 0): 'Naive(k=1)',
        (0, 1): 'Merge(k=1)',
        (1, 0): 'Naive(k=8)',
        (1, 1): 'Merge(k=8)',
    }
    styles = {
        (0, 0): {"color": "#549A84", "marker": "v", "linestyle": ":"},
        (0, 1): {"color": "#549A84", "marker": "v", "linestyle": "-"},
        (1, 0): {"color": "#CE973E", "marker": "d", "linestyle": ":"},
        (1, 1): {"color": "#CE973E", "marker": "d", "linestyle": "-"},
    }
    legend_handles = []
    legend_labels = []
    for idx, index_type in enumerate(index_types):
        # tput plot
        fig, ax = _make_fixed_plot_area_figure(2, 1.3,
            include_xlabel=False,
            include_ylabel=(idx == 0))
        for key_idx, key_length in enumerate(EXP_MERGE_KEY_LENGTHS):
            for merge_idx, merge_level in enumerate(EXP_MERGE_LEVELS[index_type]):
                avg_values = _convert_mops_to_bops(tputs[((index_type, key_length, merge_level))]['avg'], index_type)
                min_values = _convert_mops_to_bops(tputs[((index_type, key_length, merge_level))]['min'], index_type)
                max_values = _convert_mops_to_bops(tputs[((index_type, key_length, merge_level))]['max'], index_type)
                tput_ydata = avg_values
                line, = ax.plot(
                    EXP_MERGE_ERASE_RATIOS[1:], tput_ydata,
                    label=labels[(key_idx, merge_idx)],
                    linewidth=2, markersize=6,
                    **styles[(key_idx, merge_idx)]
                )
                _add_throughput_error_bars(
                    ax,
                    EXP_MERGE_ERASE_RATIOS[1:],
                    avg_values,
                    min_values,
                    max_values,
                    color=styles[(key_idx, merge_idx)]['color'],
                )
                if labels[(key_idx, merge_idx)] not in legend_labels:
                    legend_handles.append(line)
                    legend_labels.append(labels[(key_idx, merge_idx)])
        ax.set_ylim(bottom=0)
        ax.set_xlim(left=0, right=1)
        if idx == 0:
            ax.set_ylabel(r'Throughput ($10^9$/s)')
        ax.grid(True, which='major', linestyle='--', linewidth=0.6, alpha=0.5)
        plt.savefig(f'{plot_file_prefix}-{plot_names[idx]}-tput.pdf', bbox_inches='tight')
        plt.close(fig)
        # space plot
        fig, ax = _make_fixed_plot_area_figure(2, 1.3,
            include_xlabel=True,
            include_ylabel=(idx == 0))
        for key_idx, key_length in enumerate(EXP_MERGE_KEY_LENGTHS):
            for merge_idx, merge_level in enumerate(EXP_MERGE_LEVELS[index_type]):
                space_ydata = spaces[(index_type, key_length, merge_level)]
                space_ydata = [s / space_ydata[0] for s in space_ydata]
                line, = ax.plot(
                    EXP_MERGE_ERASE_RATIOS, space_ydata,
                    label=labels[(key_idx, merge_idx)],
                    linewidth=2, markersize=6,
                    **styles[(key_idx, merge_idx)]
                )
                if labels[(key_idx, merge_idx)] not in legend_labels:
                    legend_handles.append(line)
                    legend_labels.append(labels[(key_idx, merge_idx)])
        ax.set_ylim(bottom=0)
        ax.set_xlim(left=0, right=1)
        ax.set_xlabel('Delete Ratio')
        if idx == 0:
            ax.set_ylabel(r'Relative Space')
        ax.grid(True, which='major', linestyle='--', linewidth=0.6, alpha=0.5)
        plt.savefig(f'{plot_file_prefix}-{plot_names[idx]}-space.pdf', bbox_inches='tight')
        plt.close(fig)
    fig, ax = plt.subplots(1, 1, figsize=(6, 0.3), constrained_layout=True)
    ax.legend(legend_handles, legend_labels, loc='center', ncol=len(legend_labels))
    ax.axis('off')
    plt.savefig(f'{plot_file_prefix}-legend.pdf', bbox_inches='tight')
    plt.close(fig)

def intro_plots(configs_and_results, plot_file_prefix):
    tputs = {}
    tree_indexes = [IndexType.cpu_art, IndexType.cpu_masstree, IndexType.gpu_masstree,]
    hashtable_indexes = [IndexType.cpu_libcuckoo, IndexType.gpu_dycuckoo, IndexType.gpu_extendhashtable,]
    plot_spec = [
        (0, tree_indexes, [ResultType.lookup, ResultType.scan, ResultType.mixed]),
        (1, hashtable_indexes, [ResultType.lookup, ResultType.mixed]),
    ]
    intro_plot_key_length = 16
    plot_names = ['tree', 'ht']
    legend_handles = []
    legend_labels = []
    for _, index_types, _ in plot_spec:
        for index_type in index_types:
            index_label = INDEX_LABELS[index_type]
            if index_label in legend_labels:
                continue
            legend_handles.append(mpatch.Patch(
                fill=False,
                edgecolor=INDEX_STYLES[index_type]['color'],
                hatch=HATCH_STYLES[index_type],
                linewidth=(2 if index_type in INDEX_TYPES_ROBUST else 1),
            ))
            legend_labels.append(index_label)
    for _, index_types, result_types in plot_spec:
        for result_type in result_types:
            for index_type in index_types:
                if index_type == IndexType.gpu_dycuckoo and result_type == ResultType.mixed:
                    tputs[(index_type, result_type)] = {
                        'avg': [0], 'min': [0], 'max': [0],
                    }
                else:
                    desired_config = {
                        ConfigType.index_type: index_type,
                        ConfigType.max_keys: DEFAULT_MAXKEY_LONG,
                        ConfigType.keylen_prefix: 0,
                        ConfigType.keylen_min: intro_plot_key_length,
                        ConfigType.keylen_max: intro_plot_key_length,
                    }
                    if result_type == ResultType.lookup:
                        desired_config[ConfigType.num_lookups] = DEFAULT_BATCH_SIZE
                    elif result_type == ResultType.scan:
                        desired_config[ConfigType.num_scans] = DEFAULT_SCAN_BATCH_SIZE
                        desired_config[ConfigType.scan_count] = DEFAULT_SCAN_COUNT
                    else:
                        desired_config[ConfigType.num_mixed] = DEFAULT_BATCH_SIZE
                        desired_config[ConfigType.mix_read_ratio] = DEFAULT_MIX_READ_RATIO
                    result = filter(configs_and_results, desired_config, result_type)
                    tputs[(index_type, result_type)] = {
                        'avg': [float(result[result_type.name]['avg'])],
                        'min': [float(result[result_type.name]['min'])],
                        'max': [float(result[result_type.name]['max'])],
                    }
    # plot
    for idx, index_types, result_types in plot_spec:
        fig, ax = _make_fixed_plot_area_figure(0.8 * len(result_types), 1.5,
            include_xlabel=False, include_ylabel=(idx == 0))
        bar_width = 0.22 if len(index_types) == 3 else 0.28
        bar_spacing = bar_width * 1.2
        group_centers = [group_idx for group_idx in range(len(result_types))]
        bar_offsets = [
            (index_idx - (len(index_types) - 1) / 2) * bar_spacing
            for index_idx in range(len(index_types))
        ]
        plot_top = 0
        for group_center, result_type in zip(group_centers, result_types):
            our_ymax = 0
            baseline_ymax = 0
            our_x = 0
            for bar_offset, index_type in zip(bar_offsets, index_types):
                avg_values = _convert_mops_to_bops(tputs[(index_type, result_type)]['avg'], index_type)
                min_values = _convert_mops_to_bops(tputs[(index_type, result_type)]['min'], index_type)
                max_values = _convert_mops_to_bops(tputs[(index_type, result_type)]['max'], index_type)
                ydata = avg_values
                plot_top = max(plot_top, max_values[0] if max_values[0] is not None else ydata[0])
                xdata = [group_center + bar_offset]
                if index_type in INDEX_TYPES_ROBUST:
                    our_ymax = max(our_ymax, ydata[0])
                    our_x = xdata[0]
                else:
                    baseline_ymax = max(baseline_ymax, ydata[0])
                if ydata[0] == 0:
                    ax.text(xdata[0], ydata[0], "X", fontsize=10, color='red', fontweight='bold', ha='center', va='bottom')
                else:
                    ax.bar(xdata, ydata,
                        width=bar_width,
                        fill=False,
                        edgecolor=INDEX_STYLES[index_type]['color'],
                        hatch=HATCH_STYLES[index_type],
                        linewidth=(2 if index_type in INDEX_TYPES_ROBUST else 1)
                    )
                    _add_throughput_error_bars(
                        ax,
                        xdata,
                        avg_values,
                        min_values,
                        max_values,
                        color=INDEX_STYLES[index_type]['color'],
                        for_barplot=True
                    )
            ax.text(our_x, our_ymax, f'{our_ymax / baseline_ymax:.1f}x', fontsize=12, ha='center', va='bottom')
        ax.set_ylim(bottom=0, top=plot_top * 1.15)
        xmargin = bar_spacing * len(index_types)
        ax.set_xlim(group_centers[0] - xmargin, group_centers[-1] + xmargin)
        ax.set_xticks(group_centers)
        ax.set_xticklabels([result_type.name for result_type in result_types])
        ax.grid(True, axis='y', which='major', linestyle='--', linewidth=0.6, alpha=0.5)
        if idx == 0:
            ax.set_ylabel(r'Throughput ($10^9$/s)')
        plt.savefig(f'{plot_file_prefix}-{plot_names[idx]}.pdf', bbox_inches='tight')
        plt.close(fig)
    fig, ax = plt.subplots(1, 1, figsize=(7, 0.3), constrained_layout=True)
    ax.legend(legend_handles, legend_labels, loc='center', ncol=len(legend_labels),
              handlelength=1, handletextpad=0.5)
    ax.axis('off')
    plt.savefig(f'{plot_file_prefix}-legend.pdf', bbox_inches='tight')
    plt.close(fig)

def generate_plots(args, configs_and_results):
    key_length_plots(configs_and_results, Path(args.result_dir) / 'plot_keylength')
    suffix_plots(configs_and_results, Path(args.result_dir) / 'plot_suffix')
    tile_plots(configs_and_results, Path(args.result_dir) / 'plot_tile')
    merge_plots(configs_and_results, Path(args.result_dir) / 'plot_merge')
    intro_plots(configs_and_results, Path(args.result_dir) / 'plot_intro')

if __name__ == "__main__":
    args = parse_args_for_plot()
    configs_and_results = read_configs_and_results(args)
    generate_plots(args, configs_and_results)
