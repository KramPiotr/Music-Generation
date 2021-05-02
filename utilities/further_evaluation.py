from functools import partial
import os
from statistics import mean, stdev
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from utilities.evaluation import by_model, get_sorted_keys, retrieve_, dump_, analyze_cross_corr_outsiders, \
    plot_means_outsiders
from utilities.utils import print_to_file, save_fig

evaluation_dir = "./evaluation_results"


def print_mean_stds_by_model():
    global evaluation_dir
    mean_by_model = by_model(mean)
    std_by_model = by_model(stdev)
    with open(os.path.join(evaluation_dir, "mean_std_by_model.txt"), "w") as f:
        print_ = partial(print_to_file, file=f)
        models = get_sorted_keys(mean_by_model['overall'])
        for model in models:
            print_(f"{model}:")
            for c in mean_by_model.keys():
                print_(f"{c}:", indent_level=1)
                print_(f"mean: {mean_by_model[c][model]:.2f} (in full: {mean_by_model[c][model]})", indent_level=2)
                print_(f"std: {std_by_model[c][model]:.2f} (in full: {std_by_model[c][model]})", indent_level=2)

def filter_and_intersect(filter_, xs, ys):
    def as_set(series_):
        return set(series_.where(filter_(series_)).dropna().index)

    return list(as_set(xs).intersection(as_set(ys)))

def novel_averse_seeking_division():
    overall_novel = retrieve_(name="overall_novel").fillna(0)
    novel_pleasant = retrieve_(name="novel_pleasant").fillna(0)
    corr = filter_and_intersect(lambda x: x > 0.5, overall_novel, novel_pleasant)
    uncorr = filter_and_intersect(lambda x: x < -0.5, overall_novel, novel_pleasant)
    dump_(name="uncorr-corr", file=(uncorr, corr))
    with open(os.path.join(evaluation_dir, "corr-uncorr.txt"), "w") as f:
        print_ = partial(print_to_file, file=f)
        print_(f"Positively correlated: {len(corr)}, negatively correlated: {len(uncorr)}")
        print_("Correlated")
        print_("Novelty-Pleasantness", indent_level=1)
        sorted_pleasant = novel_pleasant[corr].sort_values()
        for i, v in zip(sorted_pleasant.index, sorted_pleasant):
            print_(f"{i}: {v:.4f}", indent_level=2)
        print_("Novelty-Overall")
        sorted_overall = overall_novel[corr].sort_values()
        for i, v in zip(sorted_overall.index, sorted_overall):
            print_(f"{i}: {v:.4f}", indent_level=2)

        print_("Uncorrelated")
        print_("Novelty-Pleasantness", indent_level=1)
        sorted_pleasant = novel_pleasant[uncorr].sort_values()[::-1]
        for i, v in zip(sorted_pleasant.index, sorted_pleasant):
            print_(f"{i}: {v:.4f}", indent_level=2)
        print_("Novelty-Overall")
        sorted_overall = overall_novel[uncorr].sort_values()[::-1]
        for i, v in zip(sorted_overall.index, sorted_overall):
            print_(f"{i}: {v:.4f}", indent_level=2)


def plot_cross_corr_by_user():
    overall_novel = retrieve_(name="overall_novel").fillna(0)
    novel_pleasant = retrieve_(name="novel_pleasant").fillna(0)
    np.random.seed(20)
    colors = ['#EF9A9A', '#81D4FA', '#C62828', '#0277BD']
    #colors = color_list(4)
    color = np.array([colors[int(x > 0)] for x in overall_novel])
    uncorr, corr = retrieve_(name="uncorr-corr")
    color[uncorr] = colors[2]
    color[corr] = colors[3]
    plt.figure(figsize=(7, 7.5))
    #plt.figure()
    plt.scatter(overall_novel, novel_pleasant, c=color, alpha=0.7)
    plt.plot([-1, 1], [-1, 1], '--', alpha=0.3)
    corr_name_ = "Novelty-Overall"
    plt.xlabel(f"{corr_name_} correlation")
    plt.ylabel("Novelty-Pleasantness correlation")
    plt.title("Cross-criteria correlation by user")
    plt.subplots_adjust(bottom=0.2)
    #plt.tight_layout(2, rect=(0, 2, 1, 1))
    plt.grid(True)
    corr_name = "Novelty vs. Overall & Pleasantness"
    patches = [mpatches.Patch(color=colors[2], label=f'{corr_name} moderately negatively correlated'),
               mpatches.Patch(color=colors[0], label=f'{corr_name} weakly negatively correlated'),
               mpatches.Patch(color=colors[1], label=f'{corr_name} weakly positively correlated'),
               mpatches.Patch(color=colors[3], label=f'{corr_name} moderately positively correlated')]
    plt.legend(handles=patches, loc='upper center', bbox_to_anchor=(0.5, -0.1), fancybox=True, shadow=True,)
    save_fig(plt, evaluation_dir, "cross_correlation_users")


def update_db_2():
    print_mean_stds_by_model()
    novel_averse_seeking_division()
    plot_cross_corr_by_user()
    analyze_cross_corr_outsiders()
    plot_means_outsiders()


if __name__ == "__main__":
    update_db_2()
    #novel_averse_seeking_division()
