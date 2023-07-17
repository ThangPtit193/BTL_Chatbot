# -*- coding: utf-8
import itertools
import os
from collections import Counter
from typing import List, Text, Any

import numpy as np
import pandas as pd
from tabulate import tabulate
from texttable import Texttable

from comet.constants import DEFAULT_BASE_EVAL_PATH
from . import logger

_logger = logger.get_logger(__name__)

try:
    import matplotlib.pyplot as plt
except Exception as e:
    _logger.debug(f"{e}")


def print_comet(message, denver_version):
    print("")
    print('\n'.join([
        '▅ ▆ ▇ █ COMET █ ▇ ▆ ▅ {}'.format(denver_version),
        ''
    ]))


def print_style_free(message, print_fun=print):
    print_fun("")
    print_fun("░▒▓█  {}".format(message))


def print_style_time(message, print_fun=print):
    print_fun("")
    print_fun("⏰  {}".format(message))
    print_fun("")


def print_style_warning(message, print_fun=print):
    print_fun("")
    print_fun("⛔️  {}".format(message))
    print_fun("")


def print_style_notice(message, print_fun=print):
    print_fun("")
    print_fun("📌  {}".format(message))
    print_fun("")


def print_line(text, print_fun=print):
    print_fun("")
    print_fun("➖➖➖➖➖➖➖➖➖➖ {} ➖➖➖➖➖➖➖➖➖➖".format(text.upper()))
    print_fun("")


def print_boxed(text, print_fun=print):
    box_width = len(text) + 2
    print_fun('')
    print_fun('╒{}╕'.format('═' * box_width))
    print_fun('  {}  '.format(text.upper()))
    print_fun('╘{}╛'.format('═' * box_width))
    print_fun('')


def view_table(metrics, log_file_path: Text = None):
    """Function view table logger.

    :param metrics: A dict type. FORMAT: {'key1': [a, b, c], 'key2': [d, e, f]}
    :param log_file_path: Where to save the log file.
    """
    if os.path.exists(log_file_path):
        os.remove(log_file_path)

    def write_to_file(str_data):
        str_data = str(str_data)
        print(str_data)
        if log_file_path:
            with open(log_file_path, 'a') as f:
                f.write(str_data)
                f.write("\n")

    # Create a new one if file not exists
    if log_file_path is not None:
        if not os.path.exists(os.path.dirname(log_file_path)):
            os.makedirs(os.path.dirname(log_file_path))

    if "tag_detailed_results" in metrics:
        scores, by_classes = get_detail_results(metrics['tag_detailed_results'][0])
        _ = metrics.pop("tag_detailed_results", None)
        _ = metrics.update(scores)

        write_to_file(tabulate(metrics, headers="keys", tablefmt='pretty'))
        write_to_file("Detailed results: ")
        write_to_file(tabulate(by_classes, headers="keys", tablefmt='pretty'))

    elif "cls_detailed_results" in metrics:
        cls_detailed = metrics.get("cls_detailed_results")
        _ = metrics.pop("cls_detailed_results", None)

        write_to_file(tabulate(metrics, headers="keys", tablefmt='pretty'))
        if cls_detailed:
            write_to_file("Detailed results: ")
            write_to_file(cls_detailed)

    elif "intent" in metrics or "tags" in metrics:
        imetrics = metrics.get("intent")
        if imetrics:
            _ = metrics.pop("intent", None)

        cls_detailed = metrics.get("cls_detailed")
        if cls_detailed:
            _ = metrics.pop("cls_detailed", None)

        tmetrics = metrics.get("tags")
        if tmetrics:
            _ = metrics.pop("tags", None)

        tags_detailed = metrics.get("tags_detailed")
        if tags_detailed:
            _ = metrics.pop("tags_detailed", None)

        metrics = {key: value for key, value in metrics.items() if value is not None}
        if len(metrics) != 0:
            write_to_file(tabulate(metrics, headers="keys", tablefmt='pretty'))

        if imetrics:
            write_to_file("Intent results: ")
            write_to_file(tabulate(imetrics, headers="keys", tablefmt='pretty'))
        if cls_detailed:
            write_to_file("Intent detailed results: ")
            write_to_file(cls_detailed)

        if tmetrics:
            write_to_file("Tags results: ")
            write_to_file(tabulate(tmetrics, headers="keys", tablefmt='pretty'))
        if tags_detailed:
            write_to_file("Tags detailed results: ")
            write_to_file(tags_detailed)

    else:
        write_to_file(tabulate(metrics, headers="keys", tablefmt='pretty'))


def to_list(string):
    values = [t.strip() for t in string.split(' - ')]
    tag = values[0].split()[0]
    tp = " ".join(values[0].split()[1:])
    values.insert(0, "tag: " + tag)
    values[1] = tp

    return values


def get_detail_results(detail_results):
    lines = detail_results.split('\n')

    scores = [lines[2].strip('- '), lines[3].strip('- ')]
    scores = {
        'f1-score (micro)': [scores[0].split()[-1].strip()],
        'f1-score (macro)': [scores[1].split()[-1].strip()]
    }

    tags_results = [to_list(lines[i].strip()) for i in range(6, len(lines))]

    by_classes = {
        'tag': [],
        'tp': [],
        'fp': [],
        'fn': [],
        'precision': [],
        'recall': [],
        'f1-score': []
    }
    for tag_result in tags_results:
        for element in tag_result:
            temp = element.split(":")
            by_classes[temp[0].strip()].append(temp[1].strip())

    return scores, by_classes


def _plot_confusion_matrix(cm,
                           target_names,
                           title='Confusion matrix',
                           cmap=None,
                           normalize=True,
                           save_dir=None):
    """Function to plot confusion matrics.

    :param cm: confusion_matrix: function in sklearn.
    :param target_names: list of classes.
    :param cmap: str or matplotlib Colormap: Colormap recognized by matplotlib.
    :param normalize: normalizes confusion matrix over the true (rows), predicted (columns) conditions or all the population.
    :param save_dir: str: directory address to save.
    """

    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    width = int(10 / 7 * len(target_names))
    height = int(8 / 7 * len(target_names))

    plt.figure(figsize=(width, height))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=90)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.2f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="red" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="red" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel(
        'Predicted label. Metrics: accuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    try:
        _logger.debug(f"Save confusion-matrix...")
        plt.savefig((save_dir + '/{}.png'.format(title)))
    except IOError:
        _logger.error(f"Could not save file in directory: {save_dir}")


def plot_confusion_matrix(cm,
                          target_names,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=True,
                          save_dir=None):
    """Function to plot confusion matrics.

    :param cm: confusion_matrix: function in sklearn.
    :param target_names: list of classes.
    :param title: The title of the confusion matrix.
    :param cmap: str or matplotlib Colormap: Colormap recognized by matplotlib.
    :param normalize: normalizes confusion matrix over the true (rows), predicted (columns) conditions or all the population.
    :param save_dir: str: directory address to save.
    """

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        _logger.info("Normalized confusion matrix")
    else:
        _logger.info('Confusion matrix, without normalization')

    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    width = int(10 / 7 * len(target_names))
    height = int(8 / 7 * len(target_names))

    plt.figure(figsize=(width, height))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    fmt = '.2f' if normalize else 'd'
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="red" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel(
        'Predicted label. Metrics: accuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    try:
        _logger.debug(f"Save confusion-matrix...")
        plt.savefig((save_dir + '/{}.png'.format(title)))
    except IOError:
        _logger.error(f"Could not save file in directory: {save_dir}")


def print_confusion_matrix(
        cm, target_names, save_dir: Text = None, name_file: Text = None):
    """
    Prints a confusion matrix and associated metrics.
    Args:
        cm:
        target_names:
        save_dir:
        name_file:

    Returns:

    """
    save_dir = save_dir or DEFAULT_BASE_EVAL_PATH

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # name_file = name_file or "confusion_matrix.csv"

    if not name_file.endswith(".csv"):
        name_file += ".csv"

    cmtx = pd.DataFrame(
        cm,
        index=target_names,
        columns=target_names
    )
    print_line("Confusion Matrix")
    print(cmtx)
    # Write cmtx to csv file
    # cmtx.to_csv(os.path.join(save_dir, name_file))

    # Print most confused
    data_dict = cmtx.to_dict()
    most_confused = Counter()
    for _category in data_dict.keys():
        true_classes = data_dict[_category].keys()
        for _class in true_classes:
            if _class == _category:
                continue
            no = data_dict[_category][_class]

            if no == 0:
                continue

            key = f"{_category} - {_class}"
            most_confused[key] = no
    confused = [f"{x}: {most_confused[x]}" for x, _ in most_confused.most_common()]
    print_line("The most confused")
    _logger.pr("\n".join(confused), color=logger.COLOR.BLUE)


def print_title(text: Text, scale: float = 20, color: Text = "green", show: bool = True,
                trace=False):
    """
    Print title

    Args:
        text:
        scale:
        color:
        show: print to consolve
        trace: write file

    Returns: samples
        ==========================
        |         hello          |
        ==========================

    """

    # logger = LogTracker()
    width = int(len(text) + scale)
    string1 = ''.center(width, "=")
    string2 = text.center(width, " ").upper()
    string2 = "|" + string2[1:-1] + "|"
    string3 = ''.center(width, "=")

    box_text = '\n' + string1 + '\n' + string2 + '\n' + string3

    if show and not trace:
        _logger.pr(string1, color=color)
        _logger.pr(string2, color=color)
        _logger.pr(string3, color=color)
    if trace:
        _logger.trace(box_text, has_prefix=False)
    return box_text


def prints(data: List[Any]):
    """
    Print list of data
    Args:
        data:

    Returns:

    """
    # for d in data:
    #     print(str(d))
    if data is not None:
        print([str(d) for d in data])


def print_table(headers: List[Text], data: List[List], show=True, trace=False):
    """
    Print table of data

    Args:
        headers: [epoch, acc, loss]
        data:
            [[1, 2, 3], [1, 2, 3], [1, 2, 3]]
        show:
        trace:

    Returns:

    """
    assert len(headers) == len(data[0])
    tee = Texttable(max_width=180)
    tee.add_rows([headers])
    for r in data:
        tee.add_row(r)

    table_str = tee.draw()
    if show and not trace:
        print(table_str)
    if trace:
        _logger.trace(table_str, has_prefix=False, show=False)
    return table_str

# class ReliabilityDiagram(MaxProbCELoss):
#
#     def plot(self, output, labels, n_bins=15, logits=True, title=None):
#         super().loss(output, labels, n_bins, logits)
#
#         # computations
#         delta = 1.0 / n_bins
#         x = np.arange(0, 1, delta)
#         mid = np.linspace(delta / 2, 1 - delta / 2, n_bins)
#         error = np.abs(np.subtract(mid, self.bin_acc))
#
#         plt.rcParams["font.family"] = "serif"
#         # size and axis limits
#         plt.figure(figsize=(8, 8))
#         plt.xlim(0, 1)
#         plt.ylim(0, 1)
#         # plot grid
#         plt.grid(color='tab:grey', linestyle=(0, (1, 5)), linewidth=1, zorder=0)
#         # plot bars and identity line
#         plt.bar(
#             x, self.bin_acc, color='b', width=delta, align='edge',
#             edgecolor='k', label='Outputs', zorder=5
#         )
#         plt.bar(
#             x, error, bottom=np.minimum(self.bin_acc, mid), color='mistyrose',
#             alpha=0.5, width=delta, align='edge', edgecolor='r', hatch='/', label='Gap', zorder=10
#         )
#         ident = [0.0, 1.0]
#         plt.plot(ident, ident, linestyle='--', color='tab:grey', zorder=15)
#         # labels and legend
#         plt.ylabel('Accuracy', fontsize=13)
#         plt.xlabel('Confidence', fontsize=13)
#         plt.legend(loc='upper left', framealpha=1.0, fontsize='medium')
#
#         if title is not None:
#             plt.title(title, fontsize=16)
#         plt.tight_layout()
#
#         return plt
