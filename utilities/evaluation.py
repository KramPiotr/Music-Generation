from glob import glob
import os

from scipy import stats
from tqdm import tqdm
from utilities.utils import dump, retrieve, color_list, print_to_file
from statistics import mean, stdev
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import numpy as np
import pandas as pd
import re
from functools import partial
import math
import matplotlib.colors as colors
import matplotlib.cm as cmx
import copy

evaluation_dir = "./evaluation_results"
pickle_dir = os.path.join(evaluation_dir, "pickles")
os.makedirs(pickle_dir, exist_ok=True)

dump_ = partial(dump, dir=pickle_dir, extension=True)
retrieve_ = partial(retrieve, store_folder=pickle_dir, extension=True)


def save_fig(plt, store_dir, name):
    diss_eval_dir = "..\\..\\dissertation\\figs\\evaluation"
    os.makedirs(diss_eval_dir, exist_ok=True)
    from utilities.utils import save_fig
    save_fig(plt, store_dir, name)
    plt.savefig(os.path.join(diss_eval_dir, f"{name}.pdf"))

class Score():
    def rename(self, model):
        if model == "Attention":
            return "RNN_embed"
        if model == "Multihot":
            return "RNN_multihot"
        # if model == "MM":
        #     return "Markov model"
        return model

    def __init__(self, model, song, overall, pleasant, novel):
        self.model = self.rename(model)
        self.song = song
        self.overall = overall
        self.pleasant = pleasant
        self.novel = novel


def extract_score(score_line):
    splitted = score_line[0].split("/")
    model = splitted[2]
    song = splitted[3]
    overall = float(score_line[1])
    pleasant = int(score_line[2])
    novel = int(score_line[3])
    return Score(model, song, overall, pleasant, novel)


def analyze_outputs(output_dir):
    global evaluation_dir
    output_files = list(glob(os.path.join(output_dir, "output*")))
    scores = []
    comments = []
    scores_by_user = []
    for file in tqdm(output_files):
        user = []
        with open(file, "r") as f:
            output_lines = [line.split(";") for line in f.read().split("\n")]
            comment = ""
            for line in output_lines:
                if len(line) == 4 and line[0].startswith("assets/"):
                    score_ = extract_score(line)
                    user.append(score_)
                else:
                    if comment != "":
                        comment += "\n"
                    comment += ";".join(line)
            if comment != "":
                comments.append(comment)
        scores.extend(user)
        scores_by_user.append(user)
    all_comments = ("\n" + "-" * 50 + "\n").join(comments)
    os.makedirs(evaluation_dir, exist_ok=True)

    with open(os.path.join(evaluation_dir, "comments.txt"), "w") as f:
        f.write(all_comments)

    save_scores(scores, scores_by_user)


def save_scores(scores, scores_by_user):
    global evaluation_dir
    global pickle_dir
    # overall = {}
    # pleasant = {}
    # novel = {}
    #
    # overall_by_model = {}
    # pleasant_by_model = {}
    # novel_by_model = {}

    score_by_model = {}
    score_by_model_song = {}

    criteria = ["overall", "pleasant", "novel"]

    for c in criteria:
        score_by_model[c] = {}
        score_by_model_song[c] = {}

    for score in scores:
        for c in criteria:
            score_by_model[c].setdefault(score.model, [])
            score_by_model[c][score.model].append(getattr(score, c))

            score_by_model_song[c].setdefault(score.model, {})
            score_by_model_song[c][score.model].setdefault(score.song, [])
            score_by_model_song[c][score.model][score.song].append(getattr(score, c))

    dump(pickle_dir, "scores", scores, extension=True)
    dump_(name="scores_by_user", file=scores_by_user)
    dump(pickle_dir, "models", list(score_by_model["overall"].keys()), extension=True)
    dump(pickle_dir, "model_to_songs", {m: list(sts.keys()) for m, sts in score_by_model_song["overall"].items()},
         extension=True)
    dump(pickle_dir, "score_by_model", score_by_model, extension=True)
    dump(pickle_dir, "score_by_model_song", score_by_model_song, extension=True)


# def sorted_by_model(f_by_model):
#     return {c: sorted(m_s.items(), key=lambda x: x[1]) for c, m_s in f_by_model.items()}

def get_sorted_keys(dict_):
    return [key for (key, score) in sorted(dict_.items(), key=lambda x: x[1])]


def extract_label(name, model):
    if model is None:
        return name
    if model == "MusicVAE":
        return name[-8:-4]
    if model == "Bach" or model == "Random":
        return name[:-4]
    return re.search("-\d\d_\d\d_\d\d", name).group()[1:]


def in_kwargs(kwargs, arg, default):
    return default if not arg in kwargs else kwargs[arg]

def conf_95(std, n):
    return stats.t.ppf(0.975, df=n-1) * std / math.sqrt(n)

def print_means_with_conf(func):
    def wrapper(mean_by_item, std_by_item, n_items, name, **kwargs):
        global evaluation_dir
        items = get_sorted_keys(mean_by_item["overall"])
        if type(n_items) is int:
            n_items = {item: n_items for item in items}
        with open(os.path.join(evaluation_dir, f"mean_std_conf_by_model_{name}.txt"), "w") as f:
            print_ = partial(print_to_file, file=f)
            for item in items:
                print_(f"{item}:")
                for c in mean_by_item:
                    print_(f"{c}:", indent_level=1)
                    print_(f"mean: {mean_by_item[c][item]:.2f} (in full: {mean_by_item[c][item]})", indent_level=2)
                    print_(f"std: {std_by_item[c][item]:.2f} (in full: {std_by_item[c][item]})", indent_level=2)
                    conf = conf_95(std_by_item[c][item], n_items[item])
                    print_(f"95 conf: {conf:.2f} (in full: {conf})", indent_level=2)
        return func(mean_by_item, std_by_item, n_items, name, **kwargs)
    return wrapper

@print_means_with_conf
def plot_means_by_item(mean_by_item, std_by_item, n_items, name, title=None, legend=True, colours=None,
                       show=True, rotation=0, model=None, figsize=None, save=True, bar_label=False,
                       title_mp=False, **kwargs):
    global pickle_dir, evaluation_dir
    widths = [1, 2, 1]
    centers = [1, 2.5, 4]
    if colours is None:
        colours = ['#F9DC5C', '#ED254E', '#C2EABD']
    items = get_sorted_keys(mean_by_item["overall"])

    if type(n_items) is int:
        n_items = {item: n_items for item in items}

    def make_plot():
        for i, c in enumerate(["pleasant", "overall", "novel"]):
            for j, m in enumerate(items):
                bar = plt.bar(x=centers[i] + 5 * j,
                              height=mean_by_item[c][m],
                              width=widths[i],
                              color=colours[i],
                              yerr=conf_95(std_by_item[c][m], n_items[m]))
                              # error_kw={
                              #     'yerr': std_by_item[c][m],
                              #     'fmt': '.k',
                              # })
                if bar_label:
                    plt.bar_label(bar, labels=[f"{mean_by_item[c][m]:.2f}"],
                                  label_type=in_kwargs(kwargs, 'label_type', 'edge'),
                                  fontsize=in_kwargs(kwargs, 'bar_label_font_size', 10))
        plt.xticks(
            ticks=[2.5 + 5 * j for j in np.arange(len(items))],
            labels=[extract_label(n, model) for n in items]
                    if not in_kwargs(kwargs, 'remove_xticks_labels', False) else
                    [f"Song #{x}" for x in np.arange(len(items))],
            rotation=rotation
        )
        if legend:
            patches = [mpatches.Patch(color=colours[0], label='Pleasantness score'),
                       mpatches.Patch(color=colours[1], label='Overall score'),
                       mpatches.Patch(color=colours[2], label='Novelty score')]
            plt.legend(handles=patches)
        plt.ylim([0, 5])
        if title_mp and title is not None:
            plt.title(title)

    if save:
        if figsize is None:
            plt.figure()
        else:
            plt.figure(figsize=figsize)
        if title is not None:
            plt.title(title, fontdict=in_kwargs(kwargs, 'title_fontdict', dict(fontsize='large')))
        make_plot()
        save_fig(plt, evaluation_dir, name)
        if show:
            plt.show()
    return make_plot


def by_model(f, name=None):
    if name is None:
        name = "score_by_model"
    score_by_model = retrieve_(name=name)
    return {c: {m: f(l) for m, l in model_to_score.items()} for c, model_to_score in score_by_model.items()}


def by_model_song(f):
    score_by_model_song = retrieve(pickle_dir, "score_by_model_song", extension=True)
    return {
        c: {m: {s: f(l) for s, l in song_to_score.items()} for m, song_to_score in model_to_dict.items()} for
        c, model_to_dict in score_by_model_song.items()}


def std_of_stds(x):
    return math.sqrt(sum(np.array(list(x)) ** 2)) / len(x)


def accumulate_by_criterium(f, dict_):
    return {c: f(model_to_score.values()) for c, model_to_score in dict_.items()}


def switch_key_order(dict_):
    result = {}
    for c, m_s in dict_.items():
        for m, s in m_s.items():
            result.setdefault(m, {})
            result[m][c] = s
    return result


def analyze_scores(plot_model=True, name=None, representation=None, legend=True, title_mp=False, **kwargs):
    mean_by_model = by_model(mean, name)
    std_by_model = by_model(stdev, name)
    n_items = by_model(len, name)['overall']

    def transform(sentence):
        if name is None:
            return sentence
        if " " in sentence:
            return sentence + f" for {representation}"
        else:
            return sentence + f"_for_{name}"

    main_name = transform("mean_scores_by_model")
    main_title = transform("Mean scores and their 95% confidence intervals by model")

    if plot_model:
        make_main_plot = plot_means_by_item(mean_by_model, std_by_model, n_items, name=main_name,
                                            title=main_title, figsize=(10, 6.5), title_mp=title_mp, legend=legend, title_fontdict=dict(fontsize='xx-large'),
                                            **kwargs)

    mean_by_model_song = by_model_song(mean)
    std_by_model_song = by_model_song(stdev)
    len_by_model_song = by_model_song(len)
    mean_by_criteria_song = switch_key_order(mean_by_model_song)
    std_by_criteria_song = switch_key_order(std_by_model_song)

    models = get_sorted_keys(mean_by_model['overall'])
    fig = plt.figure(figsize=(12, 15))
    colours = ['#F9DC5C', '#ED254E', '#C2EABD']
    patches = [mpatches.Patch(color=colours[0], label='Pleasantness score'),
               mpatches.Patch(color=colours[1], label='Overall score'),
               mpatches.Patch(color=colours[2], label='Novelty score')]
    fig.legend(handles=patches)
    plt.suptitle(transform("Mean scores and their 95% confidence intervals by song"), fontsize=20)
    for i, m in enumerate(models):
        plt.subplot(len(models), 1, i + 1)
        plt.title(f"Music generation method: {m}, overall mean: {mean_by_model['overall'][m]:.2f}")
        make_plot = plot_means_by_item(mean_by_criteria_song[m], std_by_criteria_song[m],
                                       len_by_model_song['overall'][m],
                                       name=transform(f"mean_scores_for_{m}"),
                                       title=transform(
                                           f"Mean scores and their 95% confidence intervals by song for model {m}"),
                                       legend=False,
                                       save=False,
                                       show=False,
                                       model=m,
                                       figsize=(12, 6),
                                       rotation = 0 if m != "MusicVAE" else 15,
                                       remove_xticks_labels=True)
        make_plot()
    plt.tight_layout()
    save_fig(plt, evaluation_dir, transform("mean_scores_by_song"))
    plt.show()

    return make_main_plot


def test_re():
    print(
        re.search("-\d\d_\d\d_\d\d", "output-2021_04_17--02_44_22-None-4-1-weights-improvement-09-5.7726").group()[1:])
    print(extract_label("sdfsdf.mp3", "Bach"))


def plot_correlations(corr, name, title, rotate=True, less_ticks=False):
    def make_plot(cbar=False):
        ax = sns.heatmap(
            corr,
            vmin=-1, vmax=1, center=0,
            cmap=sns.diverging_palette(220, 20, n=200),
            square=True,
            cbar=cbar
        )

        if less_ticks:
            plt.yticks(
                ticks=ax.get_yticks()[::2],
                labels=ax.get_yticklabels()[::2]
            )

            plt.xticks(
                ticks=ax.get_xticks()[::2],
                labels=ax.get_xticklabels()[::2]
            )

        ax.set_yticklabels(
            ax.get_yticklabels(),
            rotation=0,
            horizontalalignment='right'
        )
        ax.set_xticklabels(
            ax.get_xticklabels(),
            rotation=30 if rotate else 0,
            horizontalalignment='right' if rotate else 'center'
        )

        plt.title(title)

    plt.figure(figsize=(8.2, 6.5))
    make_plot(cbar=True)
    save_fig(plt, evaluation_dir, name)
    return make_plot


def ctn(c, capitalize=True):
    '''
    criterium to name
    :return:
    '''
    if c == "pleasant":
        result = "pleasantness"
    elif c == "novel":
        result = "novelty"
    else:
        result = c
    if capitalize:
        return result.capitalize()


# def plot_cross_corr_by_user():
#     overall_novel = retrieve_(name="overall_novel").fillna(0)
#     novel_pleasant = retrieve_(name="novel_pleasant").fillna(0)
#     np.random.seed(20)
#     colors = ['#EF9A9A', '#81D4FA', '#C62828', '#0277BD']
#     #colors = color_list(4)
#     color = np.array([colors[int(x > 0)] for x in overall_novel])
#     sorted_ov_nov = overall_novel.sort_values()
#     uncorr_ov_nov = sorted_ov_nov[:10].index
#     corr_ov_nov = sorted_ov_nov[-10:].index
#     color[uncorr_ov_nov] = colors[2]
#     color[corr_ov_nov] = colors[3]
#     plt.figure()
#     plt.scatter(overall_novel, novel_pleasant, c=color, alpha=0.7)
#     plt.plot([-1, 1], [-1, 1], '--', alpha=0.3)
#     corr_name = "Novelty-Overall"
#     plt.xlabel(f"{corr_name} correlation")
#     plt.ylabel("Novelty-Pleasantness correlation")
#     plt.title("Cross-criteria correlation by user")
#     plt.grid(True)
#     patches = [mpatches.Patch(color=colors[2], label=f'{corr_name} strongly negatively correlated'),
#                mpatches.Patch(color=colors[0], label=f'{corr_name} slightly negatively correlated'),
#                mpatches.Patch(color=colors[1], label=f'{corr_name} slightly correlated'),
#                mpatches.Patch(color=colors[3], label=f'{corr_name} strongly correlated')]
#     plt.legend(handles=patches)
#     save_fig(plt, evaluation_dir, "cross_correlation_users")
#     dump_(name="uncorr-corr_ov-nov", file=(uncorr_ov_nov, corr_ov_nov))


def plot_mean_cross_correlation(magnitude=True):
    overall_novel = retrieve_(name="overall_novel")
    overall_pleasant = retrieve_(name="overall_pleasant")
    novel_pleasant = retrieve_(name="novel_pleasant")

    corrs = [overall_novel, overall_pleasant, novel_pleasant]
    abs_corrs = [x.abs() for x in corrs]

    plt.figure()
    if magnitude:
        plt.title("Mean magnitudes of the correlations and their 95% confidence intervals")
        ymaterial = abs_corrs
        filename = "cross_correlation_magnitude"
    else:
        plt.title("Mean correlations and their 95% confidence intervals")
        ymaterial = corrs
        filename = "cross_correlation"

    xs = np.arange(3)

    ys = [x.mean() for x in ymaterial]
    yerrs = [conf_95(x.std(), len(x)) for x in ymaterial]

    labels = ["Overall-Novelty", "Overall-Pleasantness", "Novelty-Pleasantness"]
    np.random.seed(10)

    def make_plot():
        plt.ylim([-1.15, 1.15])
        bars = plt.bar(x=xs, height=ys, yerr=yerrs,
                       color='#ED254E')  # color=['#F9DC5C', '#ED254E', '#C2EABD'])#color_list(xs, 3))
        plt.xticks(xs, labels)
        plt.bar_label(bars, labels=[f"{y:.2f}" for y in ys])
        # for x in xs:
        #     plt.text()

    make_plot()
    save_fig(plt, evaluation_dir, filename)
    return make_plot


def calculate_cross_criteria_correlation():
    score_dataframe = retrieve_(name="score_dataframe")
    overall_novel = score_dataframe["overall"].corrwith(score_dataframe["novel"], axis=1)
    overall_pleasant = score_dataframe["overall"].corrwith(score_dataframe["pleasant"], axis=1)
    novel_pleasant = score_dataframe["novel"].corrwith(score_dataframe["pleasant"], axis=1)
    dump_(name="overall_novel", file=overall_novel)
    dump_(name="overall_pleasant", file=overall_pleasant)
    dump_(name="novel_pleasant", file=novel_pleasant)


# TODO uporzadkuj i zaplotuj dla tych grup uzytkownikow

def multiplot_correlation(make_plots):
    plt.figure(figsize=(13.3, 8))
    plt.suptitle("Correlations for scores by model and participant", fontsize=20)

    order = [1, 4, 2, 5, 3, 6]
    for o, mp in zip(order, make_plots):
        plt.subplot(2, 3, o)
        mp()
    plt.tight_layout(2)

    norm = colors.Normalize(vmin=-1, vmax=1)
    cmap = sns.diverging_palette(220, 20, n=200, as_cmap=True)
    scalar_map = cmx.ScalarMappable(norm=norm, cmap=cmap)
    plt.subplots_adjust(bottom=0.1, right=0.9, top=0.9)
    cax = plt.axes([0.94, 0.1, 0.02, 0.8])
    cb = plt.colorbar(scalar_map, cax=cax)
    cb.outline.set_linewidth(0)

    save_fig(plt, evaluation_dir, "multiplot_correlation")
    plt.show()


def transform_and_plot_correlation():
    score_by_model = retrieve_(name="score_by_model")
    # scores_by_user = retrieve_(name="scores_by_user")
    corr_by_model = {}
    corr_by_user = {}

    make_plots = []
    raw_score_dataframe = copy.deepcopy(score_by_model)
    for c in score_by_model.keys():
        for m in score_by_model[c].keys():
            score_by_model[c][m] = np.array(score_by_model[c][m])
            score_by_model[c][m] = (score_by_model[c][m][::2] + score_by_model[c][m][1::2]) / 2
        score_by_model[c] = pd.DataFrame(score_by_model[c])
        raw_score_dataframe[c] = pd.DataFrame(raw_score_dataframe[c])
        corr_by_model[c] = score_by_model[c].corr()
        corr_by_user[c] = score_by_model[c].T.corr()
        mp = plot_correlations(corr_by_model[c], f"model_correlation_{c}", f"{ctn(c)} scores correlated by model")
        make_plots.append(mp)
        mp = plot_correlations(corr_by_user[c], f"user_correlation_{c}",
                               f"{ctn(c)} scores correlated by participant", rotate=False, less_ticks=True)
        make_plots.append(mp)
    multiplot_correlation(make_plots)
    dump_(name="score_dataframe", file=score_by_model)
    dump_(name="raw_score_dataframe", file=raw_score_dataframe)
    dump_(name="corr_by_model", file=corr_by_model)
    dump_(name="corr_by_user", file=corr_by_user)
    raw_uncorr_corr()
    # KMeans()

def raw_uncorr_corr():
    uncorr_corr = list(retrieve_(name="uncorr-corr"))
    raw_score_dataframe = retrieve_(name="raw_score_dataframe")
    raw_uncorr_corr = [np.array([(2*x, 2*x + 1) for x in xs]).flatten() for xs in uncorr_corr]
    raw_dfs = [{}, {}]
    for i, ind in enumerate(raw_uncorr_corr):
        for c in raw_score_dataframe.keys():
            raw_dfs[i][c] = raw_score_dataframe[c].iloc[ind]
    dump_(name="raw_uncorr_df", file=raw_dfs[0])
    dump_(name="raw_corr_df", file=raw_dfs[1])



def analyze_cross_corr_outsiders():
    uncorr_ind, corr_ind = retrieve_(name="uncorr-corr")
    score_dataframe = retrieve_(name="score_dataframe")
    dataframes = [{}, {}]
    for i, ind in enumerate([uncorr_ind, corr_ind]):
        for c in score_dataframe.keys():
            dataframes[i][c] = score_dataframe[c].iloc[ind]
    dump_(name="uncorr_df", file=dataframes[0])
    dump_(name="corr_df", file=dataframes[1])
    mps = [analyze_scores(plot_model=True, name="uncorr_df", representation="novelty averse people",
                          title_mp=True, legend=False, bar_label=True, label_type='center', bar_label_font_size=10),
           analyze_scores(plot_model=True, name="corr_df", representation="novelty seeking people",
                          title_mp=True, bar_label=True, label_type='center', bar_label_font_size=10)]
    plt.figure(figsize=(10, 8))
    plt.subplot(2, 1, 1)
    mps[1]()
    plt.subplot(2, 1, 2)
    mps[0]()
    plt.tight_layout()
    save_fig(plt, evaluation_dir, "multiplot_novelty")
    plt.show()

# def len_df(df=None):
#     if df is None:
#         df = by_model(len, name)['overall']
#     return list(df.values())[0].shape[0]

def plot_means_outsiders():
    dfs = ["raw_uncorr_df", "raw_corr_df"]
    keys = ["Averse", "Seeking"]
    mean_by_item = {}
    std_by_item = {}
    lens = {}
    for df, k in zip(dfs, keys):
        means = accumulate_by_criterium(mean, by_model(mean, df))
        stds = accumulate_by_criterium(std_of_stds, by_model(stdev, df))
        lens[k] = list(retrieve_(name=df).values())[0].shape[0]
        for c in means.keys():
            mean_by_item.setdefault(c, {})
            std_by_item.setdefault(c, {})
            mean_by_item[c][k] = means[c]
            std_by_item[c][k] = stds[c]
    plot_means_by_item(mean_by_item, std_by_item, lens, "mean_novelty",
                       title="Mean scores and their 95% confidence intervals depending on the attitude to novelty",
                       figsize=(8, 6), bar_label=True)


def process_cross_correlations():
    calculate_cross_criteria_correlation()
    plot_mean_cross_correlation()
    plot_mean_cross_correlation(False)


def process_all_correlations():
    transform_and_plot_correlation()
    process_cross_correlations()
    # the latter goes to further_evaluation
    # plot_cross_corr_by_user()
    # analyze_cross_corr_outsiders()
    # plot_means_outsiders()


def update_database(db):
    global evaluation_dir
    analyze_outputs(db)
    analyze_scores(plot_model=True, bar_label=True)
    process_all_correlations()


def change_rc_params():
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["errorbar.capsize"] = 4
    update_database(db="./output_backup_2904")


def test_accumulate():
    m1 = accumulate_by_criterium(mean, by_model(mean, "uncorr_df"))
    m2 = accumulate_by_criterium(mean, by_model(mean, "raw_uncorr_df"))
    s1 = accumulate_by_criterium(std_of_stds, by_model(stdev, "uncorr_df"))
    s2 = accumulate_by_criterium(std_of_stds, by_model(stdev, "raw_uncorr_df"))
    assert m1 == m2
    assert s1 != s2

if __name__ == "__main__":
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["errorbar.capsize"] = 4

    update_database(db="./output_backup_2904")
    # test_accumulate()
    # analyze_cross_corr_outsiders()
    # raw_uncorr_corr()
    # plot_mean_cross_correlation(False)
    # change_rc_params()
    # analyze_cross_corr_outsiders()
    # update_database(db="./output_backup_2904")
    # transform_and_plot_correlation()
    # score_dataframe = retrieve_(name="score_dataframe")
    # process_all_correlations()
    # plot_cross_corr_by_user()
    # plot_cross_correlations()
    # cross_criteria_correlation()
    # correlation()
    # test_re()
    # analyze_outputs(output_dir="./evaluation_output_backup")
    # analyze_scores(plot_model=False)
    # analyze_outputs("./output_backup_2504")
    # update_database(db="./output_backup_2504")
