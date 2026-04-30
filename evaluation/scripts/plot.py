from evaluate import *
from constants import *
import matplotlib.legend_handler as mlegh
import matplotlib.lines as mline
import matplotlib.patches as mpatch
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import statistics

GPU_VM_HOURLY_PRICE = 3.673
CPU_VM_HOURLY_PRICE = 3.648
#CPU_BASELINE_ADJUST = GPU_VM_HOURLY_PRICE / CPU_VM_HOURLY_PRICE
CPU_BASELINE_ADJUST = 1

INDEX_LABELS = {
    IndexType.gpu_masstree: "GPUMasstree",
    IndexType.gpu_chainhashtable: "GPUChainHT",
    IndexType.gpu_cuckoohashtable: "GPUCuckooHT",
    IndexType.gpu_extendhashtable: "GPUExtendHT",
    IndexType.gpu_blink_tree: "GPUBtree",
    IndexType.gpu_dycuckoo: "DyCuckoo",
    IndexType.cpu_art: "ART",
    IndexType.cpu_masstree: "Masstree",
    IndexType.cpu_libcuckoo: "libcuckoo",
    IndexType.cpu_onetbb: "oneTBB",
}
INDEX_STYLES = {
    IndexType.gpu_masstree: {"color": "#0B6E4F", "marker": "o", "linestyle": "-"},
    IndexType.gpu_chainhashtable: {"color": "#D1495B", "marker": "s", "linestyle": "-"},
    IndexType.gpu_cuckoohashtable: {"color": "#00798C", "marker": "^", "linestyle": "-"},
    IndexType.gpu_extendhashtable: {"color": "#EDAE49", "marker": "D", "linestyle": "-"},
    IndexType.gpu_blink_tree: {"color": "#9C6644", "marker": "<", "linestyle": ":"},
    IndexType.gpu_dycuckoo: {"color": "#3B60E4", "marker": "h", "linestyle": ":"},
    IndexType.cpu_art: {"color": "#5C4D7D", "marker": "P", "linestyle": "--"},
    IndexType.cpu_masstree: {"color": "#6C9A8B", "marker": "X", "linestyle": "--"},
    IndexType.cpu_libcuckoo: {"color": "#8F2D56", "marker": "v", "linestyle": "--"},
    IndexType.cpu_onetbb: {"color": "#2A9D8F", "marker": "8", "linestyle": "--"},
}
HATCH_STYLES = {
    IndexType.gpu_masstree: 'oo',
    IndexType.gpu_cuckoohashtable: '**',
    IndexType.gpu_chainhashtable: '++',
    IndexType.gpu_extendhashtable: 'xx',
    IndexType.cpu_art: '///',
    IndexType.cpu_masstree: '\\\\\\',
    IndexType.cpu_libcuckoo: '...',
    IndexType.cpu_onetbb: '|||',
}
EXTRA_COLORS = [
    "#3B60E4", "#00798C", "#8F2D56", "#0B6E4F", "#EDAE49", "#D1495B",
]

def _convert_mops_to_bops(values, index_type):
    bops = [None if v is None else v / 1000 for v in values]
    if index_type in INDEX_TYPES_CPU_BASELINE:
        bops = [v * CPU_BASELINE_ADJUST for v in bops]
    return bops

def _compute_avg_min_max_from_raw(raw_values):
    assert len(raw_values) >= 10
    average = sum(raw_values) / len(raw_values)
    percentiles = statistics.quantiles(raw_values, n=100)
    cutoff_percent = 10
    return {'avg': average,
            'min': percentiles[cutoff_percent-1],
            'max': percentiles[100-cutoff_percent-1]}

def _record_max_tput(index_type, avg_tputs, max_tputs):
    for i in range(len(avg_tputs)):
        if max_tputs[i] is None or avg_tputs[i] > max_tputs[i][0]:
            max_tputs[i] = (avg_tputs[i], index_type.name)

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

def _save_grouped_legend_pdf(legend_groups, output_file):
    group_order = ['ours', 'gpu_baseline', 'cpu_baseline']
    group_names = {
        'ours': 'This Paper',
        'gpu_baseline': 'GPU Baselines',
        'cpu_baseline': 'CPU Baselines',
    }
    ncols = 2
    group_rows = {
        group: max(1, (len(legend_groups[group]['labels']) + ncols - 1) // ncols)
        for group in group_order
    }
    legend_width = 2.95
    legend_height = legend_width * 2.3 / 3
    fig, axes = plt.subplots(3, 1,
        figsize=(legend_width, legend_height),
        gridspec_kw={
            'height_ratios': [
                group_rows[group] + (0.95 if group_rows[group] == 1 else 0.55)
                for group in group_order
            ]
        },
    )
    fig.subplots_adjust(left=0.05, right=0.95, bottom=0.025, top=0.975, hspace=0.02)
    for ax, group_name in zip(axes, group_order):
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        frame = mpatch.FancyBboxPatch(
            (0.01, 0.03), 0.98, 0.94,
            boxstyle='round,pad=0.012,rounding_size=0.02',
            linewidth=1,
            edgecolor='#cccccc',
            facecolor='white',
            transform=ax.transAxes,
            clip_on=False,
        )
        ax.add_patch(frame)
        rows = group_rows[group_name]
        ax.text(
            0.5, 0.78 if rows == 1 else 0.86, group_names[group_name],
            ha='center', va='center',
            fontsize=10, fontweight='bold',
            transform=ax.transAxes,
        )
        row_centers = [0.36] if rows == 1 else [
            0.54 - row * (0.34 / (rows - 1)) for row in range(rows)
        ]
        col_centers = [0.25, 0.75]
        for idx, (handle, label) in enumerate(zip(
            legend_groups[group_name]['handles'],
            legend_groups[group_name]['labels'],
        )):
            row = idx % rows
            col = idx // rows
            x = col_centers[col]
            if rows == 1:
                line_y = 0.53
                text_y = 0.20
            else:
                line_y = row_centers[row] + 0.085
                text_y = row_centers[row] - 0.085
            ax.plot(
                [x - 0.15, x, x + 0.15],
                [line_y, line_y, line_y],
                color=handle.get_color(),
                linestyle=handle.get_linestyle(),
                linewidth=handle.get_linewidth(),
                marker=handle.get_marker(),
                markersize=handle.get_markersize(),
                markerfacecolor=handle.get_markerfacecolor(),
                markeredgecolor=handle.get_markeredgecolor(),
                markevery=[1],
                transform=ax.transAxes,
                clip_on=False,
            )
            ax.text(
                x, text_y, label,
                ha='center', va='center',
                fontsize=10,
                transform=ax.transAxes,
            )
    fig.savefig(output_file, bbox_inches='tight', pad_inches=0.02)
    plt.close(fig)

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
                    ConfigType.valuelen_min: DEFAULT_VALUE_LENGTH_OVERVIEW,
                    ConfigType.valuelen_max: DEFAULT_VALUE_LENGTH_OVERVIEW,
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
                processed_result = _compute_avg_min_max_from_raw(result[result_type.name]['raw'])
                for metric_type in ['avg', 'min', 'max']:
                    tputs[index_type][result_type][metric_type].append(processed_result[metric_type])
    # plot
    key_lengths_bytes = [4 * l for l in EXP_KEY_LENGTHS]
    legends = {
        'ours': {'handles': [], 'labels': []},
        'gpu_baseline': {'handles': [], 'labels': []},
        'cpu_baseline': {'handles': [], 'labels': []},
    }
    def get_index_group(index_type):
        if index_type in INDEX_TYPES_ROBUST:
            return 'ours'
        if index_type in INDEX_TYPES_GPU_BASELINE:
            return 'gpu_baseline'
        if index_type in INDEX_TYPES_CPU_BASELINE:
            return 'cpu_baseline'
        assert False
    tree_indexes = [i for i in all_index_types if i in IS_INDEX_TYPE_ORDERED]
    hashtable_indexes = [i for i in all_index_types if i not in IS_INDEX_TYPE_ORDERED]
    plot_spec = [
        (tree_indexes, ResultType.lookup, False, True),
        (tree_indexes, ResultType.insert, False, False),
        (tree_indexes, ResultType.delete, False, False),
        (tree_indexes, ResultType.mixed, False, False),
        (tree_indexes, ResultType.scan, False, False),
        (hashtable_indexes, ResultType.lookup, True, True),
        (hashtable_indexes, ResultType.insert, True, False),
        (hashtable_indexes, ResultType.delete, True, False),
        (hashtable_indexes, ResultType.mixed, True, False),
    ]
    plot_names = [
        'tree-lookup', 'tree-insert', 'tree-delete', 'tree-mixed', 'tree-scan',
        'ht-lookup', 'ht-insert', 'ht-delete', 'ht-mixed'
    ]
    for idx, (index_types, result_type, set_xlabel, set_ylabel) in enumerate(plot_spec):
        fig, ax = _make_fixed_plot_area_figure(2, 1.3,
            include_xlabel=set_xlabel,
            include_ylabel=set_ylabel,
        )
        our_max = [None for _ in range(len(EXP_KEY_LENGTHS))]
        gpu_baseline_max = [None for _ in range(len(EXP_KEY_LENGTHS))]
        cpu_baseline_max = [None for _ in range(len(EXP_KEY_LENGTHS))]
        for index_type in index_types:
            if result_type not in tputs[index_type]:
                continue
            avg_values = _convert_mops_to_bops(tputs[index_type][result_type]['avg'], index_type)
            min_values = _convert_mops_to_bops(tputs[index_type][result_type]['min'], index_type)
            max_values = _convert_mops_to_bops(tputs[index_type][result_type]['max'], index_type)
            if index_type in INDEX_TYPES_ROBUST:
                _record_max_tput(index_type, avg_values, our_max)
            elif index_type in INDEX_TYPES_GPU_BASELINE:
                _record_max_tput(index_type, avg_values, gpu_baseline_max)
            else:
                _record_max_tput(index_type, avg_values, cpu_baseline_max)
            ydata = avg_values.copy()
            markevery = range(len(ydata))
            if len(markevery) == 1:
                ydata.append(0)
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
                ax.text(xdata[1], ydata[1], "X", fontsize=10, color='red', fontweight='bold', ha='center', va='center', zorder=10)
            #if index_type in IS_INDEX_TYPE_ORDERED:
            #    index_label = f'[Tree] {index_label}'
            #else:
            #    index_label = f'[HT] {index_label}'
            if index_label not in legends[get_index_group(index_type)]['labels']:
                legends[get_index_group(index_type)]['labels'].append(index_label)
                legends[get_index_group(index_type)]['handles'].append(line)
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
        for i in range(len(EXP_KEY_LENGTHS)):
            speedup_over_cpu = our_max[i][0] / cpu_baseline_max[i][0]
            speedup_over_gpu = our_max[i][0] / gpu_baseline_max[i][0] if gpu_baseline_max[i] is not None else 0
            print(f'{plot_names[idx]}: key={EXP_KEY_LENGTHS[i]} ' + \
                  f'over cpu: {speedup_over_cpu:.1f} ({our_max[i][1]}, {cpu_baseline_max[i][1]})' + \
                  f'over gpu: {speedup_over_gpu:.1f}')
    _save_grouped_legend_pdf(legends, f'{plot_file_prefix}-legend.pdf')

def key_length_cpu_plots(configs_and_results, plot_file_prefix):
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
                    ConfigType.valuelen_min: DEFAULT_VALUE_LENGTH_OVERVIEW,
                    ConfigType.valuelen_max: DEFAULT_VALUE_LENGTH_OVERVIEW,
                }
                if index_type in INDEX_TYPES_ROBUST + INDEX_TYPES_GPU_BASELINE:
                    desired_config[ConfigType.use_pinned_host_memory] = 1
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
                processed_result = _compute_avg_min_max_from_raw(result[result_type.name]['raw'])
                for metric_type in ['avg', 'min', 'max']:
                    tputs[index_type][result_type][metric_type].append(processed_result[metric_type])
    # plot
    key_lengths_bytes = [4 * l for l in EXP_KEY_LENGTHS]
    tree_indexes = [i for i in all_index_types if i in IS_INDEX_TYPE_ORDERED]
    hashtable_indexes = [i for i in all_index_types if i not in IS_INDEX_TYPE_ORDERED]
    plot_spec = [
        [
            (tree_indexes, ResultType.lookup),
            (tree_indexes, ResultType.scan),
            (tree_indexes, ResultType.insert),
            (tree_indexes, ResultType.delete),
            (tree_indexes, ResultType.mixed),
        ],
        [
            (hashtable_indexes, ResultType.lookup),
            (hashtable_indexes, ResultType.insert),
            (hashtable_indexes, ResultType.delete),
            (hashtable_indexes, ResultType.mixed),
        ],
    ]
    plot_names = ['tree', 'ht']
    legend_indexes = set()
    legend_handles = []
    legend_labels = []
    for idx, plots in enumerate(plot_spec):
        figwidth = 6
        figheight = 1.8 if idx == 0 else 2.0
        fig, axes = plt.subplots(1, len(plots), figsize=(figwidth, figheight), constrained_layout=True)
        for subplot_idx, (index_types, result_type) in enumerate(plots):
            ymax = 0
            our_max = [None for _ in range(len(EXP_KEY_LENGTHS))]
            gpu_baseline_max = [None for _ in range(len(EXP_KEY_LENGTHS))]
            cpu_baseline_max = [None for _ in range(len(EXP_KEY_LENGTHS))]
            for index_type in index_types:
                if result_type not in tputs[index_type]:
                    continue
                avg_values = _convert_mops_to_bops(tputs[index_type][result_type]['avg'], index_type)
                min_values = _convert_mops_to_bops(tputs[index_type][result_type]['min'], index_type)
                max_values = _convert_mops_to_bops(tputs[index_type][result_type]['max'], index_type)
                if index_type in INDEX_TYPES_ROBUST:
                    _record_max_tput(index_type, avg_values, our_max)
                elif index_type in INDEX_TYPES_GPU_BASELINE:
                    _record_max_tput(index_type, avg_values, gpu_baseline_max)
                else:
                    _record_max_tput(index_type, avg_values, cpu_baseline_max)
                ydata = avg_values.copy()
                ymax = max(ymax, max(max_values))
                markevery = range(len(ydata))
                if len(markevery) == 1:
                    ydata.append(0)
                xdata = key_lengths_bytes[0:len(ydata)]
                line, = axes[subplot_idx].plot(
                    xdata, ydata,
                    label=INDEX_LABELS[index_type],
                    markevery=markevery,
                    linewidth=2, markersize=6,
                    **INDEX_STYLES[index_type]
                )
                _add_throughput_error_bars(
                    axes[subplot_idx], key_lengths_bytes, avg_values, min_values, max_values, color=INDEX_STYLES[index_type]['color']
                )
                if len(markevery) == 1:
                    axes[subplot_idx].text(xdata[1], ydata[1], "X", fontsize=10, color='red', fontweight='bold', ha='center', va='center', zorder=10)
                if index_type not in legend_indexes:
                    legend_indexes.add(index_type)
                    legend_labels.append(INDEX_LABELS[index_type])
                    legend_handles.append(line)
            axes[subplot_idx].set_ylim(bottom=0, top=ymax * 1.1)
            axes[subplot_idx].set_xlim(left=0)
            _, ymax = axes[subplot_idx].get_ylim()
            ytick_candidates = [0.1, 0.2, 0.5, 1.0]
            for ytick in ytick_candidates:
                num_ticks = int(ymax // ytick)
                if 1 <= num_ticks and num_ticks < 4:
                    yticks = [ytick * x for x in range(num_ticks + 1)]
                    break
            axes[subplot_idx].set_yticks(yticks)
            axes[subplot_idx].yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))
            axes[subplot_idx].grid(True, which='major', linestyle='--', linewidth=0.6, alpha=0.5)
            axes[subplot_idx].set_title(result_type.name, fontsize=10)
            if subplot_idx == 0:
                if idx == 0:
                    axes[subplot_idx].set_ylabel('Tree Indexes\n' + r'Throughput ($10^9$/s)')
                else:
                    axes[subplot_idx].set_ylabel('Hash Table Indexes\n' + r'Throughput ($10^9$/s)')
            for i in range(len(EXP_KEY_LENGTHS)):
                speedup_over_cpu = our_max[i][0] / cpu_baseline_max[i][0]
                speedup_over_gpu = our_max[i][0] / gpu_baseline_max[i][0] if gpu_baseline_max[i] is not None else 0
                print(f'{plot_names[idx]}-{result_type.name}-cpu: key={EXP_KEY_LENGTHS[i]} ' + \
                      f'over cpu: {speedup_over_cpu:.1f} ({our_max[i][1]}, {cpu_baseline_max[i][1]})' + \
                      f'over gpu: {speedup_over_gpu:.1f}')
        if idx == 1:
            fig.supxlabel('Key Length (B)', fontsize=10)
        fig.savefig(f'{plot_file_prefix}-{plot_names[idx]}.pdf', bbox_inches='tight')
        plt.close(fig)
    fig, ax = plt.subplots(1, 1, figsize=(7, 0.5), constrained_layout=True)
    ax.legend(legend_handles, legend_labels, loc='center', ncol=len(legend_labels) / 2, handlelength=2.5)
    ax.axis("off")
    plt.savefig(f'{plot_file_prefix}-legend.pdf', bbox_inches='tight')
    plt.close(fig)

def average_slowdown_cpu(configs_and_results):
    slowdowns = []
    all_index_types = INDEX_TYPES_ROBUST
    for index_type in all_index_types:
        result_types = [ResultType.lookup, ResultType.insert, ResultType.delete]
        result_types.append(ResultType.mixed)
        if index_type in IS_INDEX_TYPE_ORDERED:
            result_types.append(ResultType.scan)
        key_lengths = EXP_KEY_LENGTHS
        for result_type in result_types:
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
                desired_config[ConfigType.use_pinned_host_memory] = 0
                result = filter(configs_and_results, desired_config, result_type)
                tput_gpu_input = float(result[result_type.name]['avg'])
                desired_config[ConfigType.use_pinned_host_memory] = 1
                result = filter(configs_and_results, desired_config, result_type)
                tput_cpu_input = float(result[result_type.name]['avg'])
                slowdown = tput_gpu_input / tput_cpu_input
                slowdowns.append(slowdown)
    avg_slowdown = sum(slowdowns) / len(slowdowns)
    print(f'average slowdown for cpu inputs: {avg_slowdown}')

def value_length_plots(configs_and_results, plot_file_prefix):
    tputs = {}
    for index_type in INDEX_TYPES_ROBUST:
        tputs[index_type] = {}
        result_types = [ResultType.lookup]
        if index_type in IS_INDEX_TYPE_ORDERED:
            result_types.append(ResultType.scan)
        for result_type in result_types:
            tputs[index_type][result_type] = {
                'avg': [], 'min': [], 'max': []
            }
            for value_length in EXP_VALUE_LENGTHS:
                desired_config = {
                    ConfigType.index_type: index_type,
                    ConfigType.max_keys: DEFAULT_MAXKEY_LONG,
                    ConfigType.keylen_prefix: 0,
                    ConfigType.keylen_min: 1,
                    ConfigType.keylen_max: 1,
                    ConfigType.valuelen_min: value_length,
                    ConfigType.valuelen_max: value_length,
                }
                if result_type == ResultType.lookup:
                    desired_config[ConfigType.num_lookups] = DEFAULT_BATCH_SIZE
                elif result_type == ResultType.scan:
                    desired_config[ConfigType.num_scans] = DEFAULT_SCAN_BATCH_SIZE
                    desired_config[ConfigType.scan_count] = DEFAULT_SCAN_COUNT
                result = filter(configs_and_results, desired_config, result_type)
                processed_result = _compute_avg_min_max_from_raw(result[result_type.name]['raw'])
                for metric_type in ['avg', 'min', 'max']:
                    tputs[index_type][result_type][metric_type].append(processed_result[metric_type])
    # plot
    value_lengths_bytes = [4 * l for l in EXP_VALUE_LENGTHS]
    tree_indexes = [i for i in INDEX_TYPES_ROBUST if i in IS_INDEX_TYPE_ORDERED]
    hashtable_indexes = [i for i in INDEX_TYPES_ROBUST if i not in IS_INDEX_TYPE_ORDERED]
    plot_spec = [
        (tree_indexes, ResultType.lookup),
        (tree_indexes, ResultType.scan),
        (hashtable_indexes, ResultType.lookup),
    ]
    plot_names = [
        'tree-lookup', 'tree-scan', 'ht-lookup'
    ]
    legend_indexes = set()
    legend_handles = []
    legend_labels = []
    for idx, (index_types, result_type) in enumerate(plot_spec):
        fig, ax = _make_fixed_plot_area_figure(1.3, 1.1,
            include_xlabel=True,
            include_ylabel=(idx == 0),
        )
        for index_type in index_types:
            avg_values = _convert_mops_to_bops(tputs[index_type][result_type]['avg'], index_type)
            min_values = _convert_mops_to_bops(tputs[index_type][result_type]['min'], index_type)
            max_values = _convert_mops_to_bops(tputs[index_type][result_type]['max'], index_type)
            line, = ax.plot(
                value_lengths_bytes, avg_values,
                label=INDEX_LABELS[index_type],
                linewidth=2, markersize=6,
                **INDEX_STYLES[index_type]
            )
            _add_throughput_error_bars(
                ax,
                value_lengths_bytes,
                avg_values,
                min_values,
                max_values,
                color=INDEX_STYLES[index_type]['color']
            )
            if index_type not in legend_indexes:
                legend_indexes.add(index_type)
                legend_labels.append(INDEX_LABELS[index_type])
                legend_handles.append(line)
        ax.set_ylim(bottom=0)
        ax.set_xlim(left=0)
        ax.grid(True, which='major', linestyle='--', linewidth=0.6, alpha=0.5)
        ax.set_xlabel('Value Length (B)')
        if idx == 0:
            ax.set_ylabel(r'Throughput ($10^9$/s)')
        fig.savefig(f'{plot_file_prefix}-{plot_names[idx]}.pdf', bbox_inches='tight')
        plt.close(fig)
    fig, ax = plt.subplots(1, 1, figsize=(6, 0.3), constrained_layout=True)
    ax.legend(legend_handles, legend_labels, loc='center', ncol=len(legend_labels))
    ax.axis("off")
    plt.savefig(f'{plot_file_prefix}-legend.pdf', bbox_inches='tight')
    plt.close(fig)

def suffix_plots(configs_and_results, plot_file_prefix):
    tputs = {}
    plot_specs = [
        (IndexType.gpu_masstree, ResultType.lookup, EXP_GPU_MASSTREE_OPTS, DEFAULT_MAXKEY_LONG),
        (IndexType.gpu_extendhashtable, ResultType.lookup, EXP_GPU_EXTENDHT_OPTS, 2 * DEFAULT_BATCH_SIZE),
        (IndexType.gpu_extendhashtable, ResultType.insert, EXP_GPU_EXTENDHT_OPTS, 2 * DEFAULT_BATCH_SIZE),
    ]
    plot_names = [
        'mt-lookup', 'et-lookup', 'et-insert'
    ]
    for idx, (index_type, result_type, opt_configs, max_key) in enumerate(plot_specs):
        tputs[idx] = {
            'avg': [], 'min': [], 'max': []
        }
        desired_config = {
            ConfigType.index_type: index_type,
            ConfigType.max_keys: max_key,
            ConfigType.keylen_min: DEFAULT_KEY_LENGTH,
            ConfigType.keylen_max: DEFAULT_KEY_LENGTH,
            ConfigType.valuelen_min: DEFAULT_VALUE_LENGTH,
            ConfigType.valuelen_max: DEFAULT_VALUE_LENGTH,
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
                processed_result = _compute_avg_min_max_from_raw(result[result_type.name]['raw'])
                for metric_type in ['avg', 'min', 'max']:
                    tputs[idx][metric_type].append(processed_result[metric_type])
    # plot
    mt_xticklabels = ['C', 'R', 'RS']
    et_xticklabels = ['R', 'C', 'CT', 'Ct']
    for idx, (index_type, result_type, opt_configs, _) in enumerate(plot_specs):
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
                    ConfigType.valuelen_min: DEFAULT_VALUE_LENGTH,
                    ConfigType.valuelen_max: DEFAULT_VALUE_LENGTH,
                    ConfigType.num_mixed: DEFAULT_BATCH_SIZE,
                    ConfigType.mix_read_ratio: mix_read_ratio,
                }
                result = filter(configs_and_results, {**desired_config, **opt_config}, ResultType.mixed)
                processed_result = _compute_avg_min_max_from_raw(result['mixed']['raw'])
                for metric_type in ['avg', 'min', 'max']:
                    tputs[(index_type, opt_idx)][metric_type].append(processed_result[metric_type])
    # plot
    labels = ['FullWarp', 'HalfWarp', 'HalfWarp+SemiSort']
    styles = [
        {"color": EXTRA_COLORS[0], "marker": "v", "linestyle": ":"},
        {"color": EXTRA_COLORS[1], "marker": "d", "linestyle": "-"},
        {"color": EXTRA_COLORS[2], "marker": "H", "linestyle": "-"},
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
    spaces = {}
    tputs = {}
    index_types = [IndexType.gpu_masstree, IndexType.gpu_extendhashtable]
    plot_names = ['mt', 'et']
    for index_type in index_types:
        for prefix_length, key_length in EXP_MERGE_KEY_LENGTHS:
            for merge_level in [0, EXP_MAX_MERGE_LEVEL[index_type]]:
                desired_config = {
                    ConfigType.index_type: index_type,
                    ConfigType.max_keys: DEFAULT_MAXKEY_LONG,
                    ConfigType.keylen_prefix: prefix_length,
                    ConfigType.keylen_min: key_length,
                    ConfigType.keylen_max: key_length,
                    ConfigType.valuelen_min: 1,
                    ConfigType.valuelen_max: 1,
                    ConfigType.num_space: DEFAULT_BATCH_SIZE,
                    OptionalConfigType.merge_level: merge_level,
                }
                result = filter(configs_and_results, desired_config, ResultType.space)
                spaces[(index_type, prefix_length, key_length, merge_level)] = [float(v) for v in result['space']]
    for index_type in index_types:
        tputs[index_type] = {
            'avg': [], 'min': [], 'max': []
        }
        for merge_level in range(0, EXP_MAX_MERGE_LEVEL[index_type] + 1):
            if index_type == IndexType.gpu_masstree:
                if merge_level == 0:
                    continue # skip non-concurrent case
                if merge_level in EXP_GPU_MASSTREE_MERGE_SKIP_LEVEL:
                    tputs[index_type]['avg'].append(0) # takes too long
                    tputs[index_type]['min'].append(0)
                    tputs[index_type]['max'].append(0)
                    continue
            desired_config = {
                ConfigType.index_type: index_type,
                ConfigType.max_keys: DEFAULT_MAXKEY_LONG,
                ConfigType.keylen_prefix: DEFAULT_KEY_LENGTH - 1,
                ConfigType.keylen_min: DEFAULT_KEY_LENGTH,
                ConfigType.keylen_max: DEFAULT_KEY_LENGTH,
                ConfigType.valuelen_min: DEFAULT_VALUE_LENGTH,
                ConfigType.valuelen_max: DEFAULT_VALUE_LENGTH,
                ConfigType.num_insdel: DEFAULT_BATCH_SIZE,
                OptionalConfigType.merge_level: merge_level,
            }
            result = filter(configs_and_results, desired_config, ResultType.delete)
            processed_result = _compute_avg_min_max_from_raw(result['delete']['raw'])
            tputs[index_type]['avg'].append(processed_result['avg'])
            tputs[index_type]['min'].append(processed_result['min'])
            tputs[index_type]['max'].append(processed_result['max'])
    # space plot
    labels = {
        (0, 0): '4B Key (Naive)',
        (0, 1): '4B Key (Merge)',
        (1, 0): 'Common Prefix 32B Key (Naive)',
        (1, 1): 'Common Prefix 32B Key (Merge)',
        (2, 0): 'Random Prefix 32B Key (Naive)',
        (2, 1): 'Random Prefix 32B Key (Merge)',
    }
    styles = {
        (0, 0): {"color": EXTRA_COLORS[0], "marker": "v", "linestyle": ":"},
        (0, 1): {"color": EXTRA_COLORS[1], "marker": "^", "linestyle": "-"},
        (1, 0): {"color": EXTRA_COLORS[2], "marker": "d", "linestyle": ":"},
        (1, 1): {"color": EXTRA_COLORS[3], "marker": "h", "linestyle": "-"},
        (2, 0): {"color": EXTRA_COLORS[5], "marker": "o", "linestyle": ":"},
        (2, 1): {"color": EXTRA_COLORS[4], "marker": "s", "linestyle": "-"},
    }
    margin = 0.03
    legend_handles = []
    legend_labels = []
    for idx, index_type in enumerate(index_types):
        fig_width = 3 if idx == 0 else 2.6
        fig, axes = plt.subplots(1, 3, figsize=(fig_width, 1.7), constrained_layout=True)
        for key_idx, (prefix_length, key_length) in enumerate(EXP_MERGE_KEY_LENGTHS):
            for merge_idx, merge_level in enumerate([0, EXP_MAX_MERGE_LEVEL[index_type]]):
                ydata = spaces[(index_type, prefix_length, key_length, merge_level)]
                ydata = [1 - s / ydata[0] for s in ydata]
                line, = axes[key_idx].plot(
                    EXP_MERGE_ERASE_RATIOS, ydata,
                    label=labels[((key_idx, merge_idx))],
                    linewidth=2, markersize=6,
                    **styles[(key_idx, merge_idx)]
                )
                if labels[(key_idx, merge_idx)] not in legend_labels:
                    legend_labels.append(labels[(key_idx, merge_idx)])
                    legend_handles.append(line)
            axes[key_idx].set_ylim(bottom=-margin, top=1+margin)
            axes[key_idx].set_xlim(left=-margin, right=1+margin)
            axes[key_idx].grid(True, which='major', linestyle='--', linewidth=0.6, alpha=0.5)
            axes[key_idx].set_xticks([0, 0.5, 1])
            axes[key_idx].set_xticklabels(['0', '', '1'])
            axes[key_idx].set_yticks([0, 0.5, 1])
            if key_idx != 0:
                axes[key_idx].set_yticklabels([])
        if idx == 0:
            axes[0].set_ylabel('Relative\nMemory Reduction')
        axes[1].set_xlabel('Delete Ratio')
        plt.savefig(f'{plot_file_prefix}-{plot_names[idx]}-space.pdf', bbox_inches='tight')
        plt.close(fig)
    fig, ax = plt.subplots(1, 1, figsize=(7, 0.5), constrained_layout=True)
    ax.legend(
        legend_handles,
        legend_labels,
        loc='center',
        ncol=3,
    )
    ax.axis('off')
    plt.savefig(f'{plot_file_prefix}-legend.pdf', bbox_inches='tight')
    plt.close(fig)
    #tput plot
    mt_xticklabels = ['Naive', 'Merge\nNodes\n(§4.2)', '+PPM\n(§4.3)', '+Root\nCollect\n(§4.3)']
    et_xticklabels = ['Naive', 'Merge\nChains\n(§5.4)', '+Merge\nBuckets\n(§5.5)']
    for idx, index_type in enumerate(index_types):
        fig, ax = _make_fixed_plot_area_figure(0.6 * len(tputs[index_type]['avg']), 1.2,
            include_xlabel=False, include_ylabel=(idx == 0))
        avg_values = _convert_mops_to_bops(tputs[index_type]['avg'], index_type)
        min_values = _convert_mops_to_bops(tputs[index_type]['min'], index_type)
        max_values = _convert_mops_to_bops(tputs[index_type]['max'], index_type)
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
        ax.set_ylim(bottom=0)
        ax.set_xticks(xdata)
        ax.set_xticklabels(xlabel)
        ax.grid(True, axis='y', which='major', linestyle='--', linewidth=0.6, alpha=0.5)
        if idx == 0:
            ax.set_ylabel(r'Throughput ($10^9$/s)')
        plt.savefig(f'{plot_file_prefix}-{plot_names[idx]}-tput.pdf', bbox_inches='tight')
        plt.close(fig)


def intro_plots(configs_and_results, plot_file_prefix):
    tputs = {}
    tree_indexes = [IndexType.cpu_art, IndexType.cpu_masstree, IndexType.gpu_masstree,]
    hashtable_indexes = [IndexType.cpu_libcuckoo, IndexType.cpu_onetbb, IndexType.gpu_extendhashtable,]
    plot_spec = [
        (0, tree_indexes, [ResultType.lookup, ResultType.scan, ResultType.mixed]),
        (1, hashtable_indexes, [ResultType.lookup, ResultType.mixed]),
    ]
    intro_plot_key_length = 8
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
                desired_config = {
                    ConfigType.index_type: index_type,
                    ConfigType.max_keys: DEFAULT_MAXKEY_LONG,
                    ConfigType.keylen_prefix: 0,
                    ConfigType.keylen_min: intro_plot_key_length,
                    ConfigType.keylen_max: intro_plot_key_length,
                    ConfigType.valuelen_min: DEFAULT_VALUE_LENGTH_OVERVIEW,
                    ConfigType.valuelen_max: DEFAULT_VALUE_LENGTH_OVERVIEW,
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
                processed_result = _compute_avg_min_max_from_raw(result[result_type.name]['raw'])
                tputs[(index_type, result_type)] = {
                    'avg': [processed_result['avg']],
                    'min': [processed_result['min']],
                    'max': [processed_result['max']],
                }
    # plot
    for idx, index_types, result_types in plot_spec:
        fig, ax = _make_fixed_plot_area_figure(0.8 * len(result_types), 1.2,
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
        ax.set_ylim(bottom=0, top=plot_top * 1.2)
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

def meme_plots(configs_and_results, plot_file_prefix):
    tputs = {}
    tree_indexes = [IndexType.cpu_art, IndexType.cpu_masstree, IndexType.gpu_masstree,]
    hashtable_indexes = [IndexType.cpu_libcuckoo, IndexType.cpu_onetbb, IndexType.gpu_cuckoohashtable, IndexType.gpu_chainhashtable, IndexType.gpu_extendhashtable,]
    for index_type in tree_indexes + hashtable_indexes:
        tputs[index_type] = {}
        result_types = [ResultType.lookup, ResultType.insert, ResultType.delete, ResultType.mixed]
        if index_type in IS_INDEX_TYPE_ORDERED:
            result_types.append(ResultType.scan)
        for result_type in result_types:
            tputs[index_type][result_type] = {
                'avg': [], 'min': [], 'max': []
            }
            desired_config = {
                ConfigType.index_type: index_type,
                ConfigType.dataset_file: MEME_DATASET_PATH,
                ConfigType.valuelen_min: DEFAULT_VALUE_LENGTH_OVERVIEW,
                ConfigType.valuelen_max: DEFAULT_VALUE_LENGTH_OVERVIEW,
            }
            if result_type == ResultType.lookup:
                desired_config[ConfigType.num_lookups] = BATCH_SIZE_MEME
            elif result_type in [ResultType.insert, ResultType.delete]:
                desired_config[ConfigType.num_insdel] = BATCH_SIZE_MEME
            elif result_type == ResultType.mixed:
                desired_config[ConfigType.num_mixed] = BATCH_SIZE_MEME
                desired_config[ConfigType.mix_read_ratio] = DEFAULT_MIX_READ_RATIO
            elif result_type == ResultType.scan:
                desired_config[ConfigType.num_scans] = BATCH_SIZE_MEME
                desired_config[ConfigType.scan_count] = DEFAULT_SCAN_COUNT
            result = filter(configs_and_results, desired_config, result_type)
            processed_result = _compute_avg_min_max_from_raw(result[result_type.name]['raw'])
            for metric_type in ['avg', 'min', 'max']:
                tputs[index_type][result_type][metric_type].append(processed_result[metric_type])
    # plot
    plot_spec = [
        (tree_indexes, [ResultType.lookup, ResultType.scan, ResultType.insert, ResultType.delete, ResultType.mixed]),
        (hashtable_indexes, [ResultType.lookup, ResultType.insert, ResultType.delete, ResultType.mixed]),
    ]
    plot_names = ['tree', 'ht']
    legend_handles = []
    legend_labels = []
    for index_type in [IndexType.cpu_art, IndexType.cpu_masstree, IndexType.cpu_libcuckoo, IndexType.cpu_onetbb,
                       IndexType.gpu_masstree, IndexType.gpu_cuckoohashtable, IndexType.gpu_chainhashtable, IndexType.gpu_extendhashtable]:
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
    for idx, (index_types, result_types) in enumerate(plot_spec):
        fig_width = 2.7 if idx == 0 else 2.7
        fig, ax = _make_fixed_plot_area_figure(fig_width, 1.5, include_xlabel=False, include_ylabel=(idx == 0))
        bar_width = 0.22 if len(index_types) == 3 else 0.15
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
            our_x = []
            for bar_offset, index_type in zip(bar_offsets, index_types):
                avg_values = _convert_mops_to_bops(tputs[index_type][result_type]['avg'], index_type)
                min_values = _convert_mops_to_bops(tputs[index_type][result_type]['min'], index_type)
                max_values = _convert_mops_to_bops(tputs[index_type][result_type]['max'], index_type)
                ydata = avg_values
                plot_top = max(plot_top, max_values[0] if max_values[0] is not None else ydata[0])
                xdata = [group_center + bar_offset]
                if index_type in INDEX_TYPES_ROBUST:
                    our_ymax = max(our_ymax, ydata[0])
                    our_x.append(xdata[0])
                else:
                    baseline_ymax = max(baseline_ymax, ydata[0])
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
            if baseline_ymax > 0 and our_ymax > 0:
                ax.text(sum(our_x) / len(our_x), our_ymax, f'{our_ymax / baseline_ymax:.1f}x', fontsize=12, ha='center', va='bottom')
        ax.set_ylim(bottom=0, top=plot_top * 1.2)
        xmargin = bar_spacing * len(index_types)
        ax.set_xlim(group_centers[0] - xmargin, group_centers[-1] + xmargin)
        ax.set_xticks(group_centers)
        ax.set_xticklabels([result_type.name for result_type in result_types])
        ax.grid(True, axis='y', which='major', linestyle='--', linewidth=0.6, alpha=0.5)
        if idx == 0:
            ax.set_ylabel(r'Throughput ($10^9$/s)')
        plt.savefig(f'{plot_file_prefix}-{plot_names[idx]}.pdf', bbox_inches='tight')
        plt.close(fig)
    fig, ax = plt.subplots(1, 1, figsize=(5, 0.5), constrained_layout=True)
    ax.legend(legend_handles, legend_labels, loc='center', ncol=len(legend_labels)/2,
              handlelength=2, handletextpad=0.5)
    ax.axis('off')
    plt.savefig(f'{plot_file_prefix}-legend.pdf', bbox_inches='tight')
    plt.close(fig)


def generate_plots(args, configs_and_results):
    key_length_plots(configs_and_results, Path(args.result_dir) / 'plot_keylength')
    key_length_cpu_plots(configs_and_results, Path(args.result_dir) / 'plot_keylength_cpu')
    average_slowdown_cpu(configs_and_results)
    value_length_plots(configs_and_results, Path(args.result_dir) / 'plot_valuelength')
    suffix_plots(configs_and_results, Path(args.result_dir) / 'plot_suffix')
    tile_plots(configs_and_results, Path(args.result_dir) / 'plot_tile')
    merge_plots(configs_and_results, Path(args.result_dir) / 'plot_merge')
    intro_plots(configs_and_results, Path(args.result_dir) / 'plot_intro')
    meme_plots(configs_and_results, Path(args.result_dir) / 'plot_meme')

if __name__ == "__main__":
    args = parse_args_for_plot()
    configs_and_results = read_configs_and_results(args)
    generate_plots(args, configs_and_results)
