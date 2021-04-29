from glob import glob
import os
from tqdm import tqdm
from utilities.utils import dump, retrieve, save_fig, color_list
from statistics import mean, stdev
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import numpy as np
import pandas as pd
import re
from functools import partial
from sklearn.cluster import KMeans

evaluation_dir = "./evaluation_output"
pickle_dir = os.path.join(evaluation_dir, "pickles")
os.makedirs(pickle_dir, exist_ok=True)

dump_ = partial(dump, dir=pickle_dir, extension=True)
retrieve_ = partial(retrieve, store_folder=pickle_dir, extension=True)


class Score():
    def __init__(self, model, song, overall, pleasant, novel):
        self.model = model
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


def plot_means_by_item(mean_by_item, std_by_item, name, title=None, legend=True, colours=None,
                       show=True, rotation=0, model=None, figsize=None, save=True):
    global pickle_dir, evaluation_dir
    widths = [1, 2, 1]
    centers = [1, 2.5, 4]
    if colours is None:
        colours = ['#F9DC5C', '#ED254E', '#C2EABD']
    items = get_sorted_keys(mean_by_item["overall"])

    def make_plot():
        for i, c in enumerate(["pleasant", "overall", "novel"]):
            for j, m in enumerate(items):
                plt.bar(x=centers[i] + 5 * j,
                        height=mean_by_item[c][m],
                        width=widths[i],
                        color=colours[i],
                        yerr=std_by_item[c][m])
        plt.xticks(
            ticks=[2.5 + 5 * j for j in np.arange(len(items))],
            labels=[extract_label(n, model) for n in items],
            rotation=rotation
        )
        if legend:
            patches = [mpatches.Patch(color=colours[0], label='Pleasantness score'),
                       mpatches.Patch(color=colours[1], label='Overall score'),
                       mpatches.Patch(color=colours[2], label='Novelty score')]
            plt.legend(handles=patches)
        plt.ylim([0, 5])

    if save:
        if figsize is None:
            plt.figure()
        else:
            plt.figure(figsize=figsize)
        if title is not None:
            plt.title(title)
        make_plot()
        save_fig(plt, evaluation_dir, name)
        if show:
            plt.show()
    return make_plot


def by_model(f):
    score_by_model = retrieve(pickle_dir, "score_by_model", extension=True)
    return {c: {m: f(l) for m, l in model_to_score.items()} for c, model_to_score in score_by_model.items()}


def by_model_song(f):
    score_by_model_song = retrieve(pickle_dir, "score_by_model_song", extension=True)
    return {
        c: {m: {s: f(l) for s, l in song_to_score.items()} for m, song_to_score in model_to_dict.items()} for
        c, model_to_dict in score_by_model_song.items()}


def switch_key_order(dict_):
    result = {}
    for c, m_s in dict_.items():
        for m, s in m_s.items():
            result.setdefault(m, {})
            result[m][c] = s
    return result


def analyze_scores(plot_model=True):
    mean_by_model = by_model(mean)
    std_by_model = by_model(stdev)
    if plot_model:
        plot_means_by_item(mean_by_model, std_by_model, name="mean_scores_by_model",
                           title="Mean scores and their standard deviations by model")

    mean_by_model_song = by_model_song(mean)
    std_by_model_song = by_model_song(stdev)
    mean_by_criteria_song = switch_key_order(mean_by_model_song)
    std_by_criteria_song = switch_key_order(std_by_model_song)

    models = get_sorted_keys(mean_by_model['overall'])
    plt.figure(figsize=(12, 15))
    plt.suptitle("Mean scores and their standard deviations by song", fontsize=20)
    for i, m in enumerate(models):
        plt.subplot(len(models), 1, i + 1)
        plt.title(f"Model: {m}, overall mean: {mean_by_model['overall'][m]:.2f}")
        make_plot = plot_means_by_item(mean_by_criteria_song[m], std_by_criteria_song[m], name=f"mean_scores_for_{m}",
                                       title=f"Mean scores and their standard deviations by song for model {m}",
                                       legend=i == 0,
                                       save=False,
                                       show=False,
                                       model=m,
                                       figsize=(12, 6))
        make_plot()
    plt.tight_layout()
    save_fig(plt, evaluation_dir, "mean_scores_by_song")
    plt.show()


def test_re():
    print(
        re.search("-\d\d_\d\d_\d\d", "output-2021_04_17--02_44_22-None-4-1-weights-improvement-09-5.7726").group()[1:])
    print(extract_label("sdfsdf.mp3", "Bach"))


def update_database(db):
    analyze_outputs(db)
    analyze_scores(plot_model=True)


def plot_correlations(corr, name, title, rotate=True):
    def make_plot(cbar=False):
        ax = sns.heatmap(
            corr,
            vmin=-1, vmax=1, center=0,
            cmap=sns.diverging_palette(20, 220, n=200),
            square=True,
            cbar=cbar
        )
        ax.set_yticklabels(
            ax.get_yticklabels(),
            rotation=0,
            horizontalalignment='right'
        )
        if rotate:
            ax.set_xticklabels(
                ax.get_xticklabels(),
                rotation=45,
                horizontalalignment='right'
            )

    plt.figure(figsize=(7, 6))
    make_plot(cbar=True)
    plt.title(title)
    save_fig(plt, evaluation_dir, name)
    return make_plot

def ctn(c, capitalize = True):
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


def plot_mean_cross_correlation(magnitude=True):
    overall_novel = retrieve_(name="overall_novel")
    overall_pleasant = retrieve_(name="overall_pleasant")
    novel_pleasant = retrieve_(name="novel_pleasant")

    corrs = [overall_novel, overall_pleasant, novel_pleasant]
    abs_corrs = [x.abs() for x in corrs]

    plt.figure()
    if magnitude:
        plt.title("Mean magnitudes of the correlations and their standard deviations")
        ymaterial = abs_corrs
        filename = "cross_correlation_magnitude"
    else:
        plt.title("Mean correlations and their standard deviations")
        ymaterial = corrs
        filename = "cross_correlation"

    xs = np.arange(3)

    ys = [x.mean() for x in ymaterial]
    yerrs = [x.std() for x in ymaterial]

    labels = ["Overall-Novelty", "Overall-Pleasantness", "Novelty-Pleasantness"]
    np.random.seed(10)
    def make_plot():
        plt.bar(x=xs, height=ys, yerr=yerrs, color=color_list(xs, 3))
        plt.xticks(xs, labels)
    make_plot()
    save_fig(plt, evaluation_dir, filename)
    return make_plot

def cross_criteria_correlation():
    score_dataframe = retrieve_(name="score_dataframe")
    overall_novel = score_dataframe["overall"].corrwith(score_dataframe["novel"], axis=1)
    overall_pleasant = score_dataframe["overall"].corrwith(score_dataframe["pleasant"], axis=1)
    novel_pleasant = score_dataframe["novel"].corrwith(score_dataframe["pleasant"], axis=1)
    dump_(name="overall_novel", file=overall_novel)
    dump_(name="overall_pleasant", file=overall_pleasant)
    dump_(name="novel_pleasant", file=novel_pleasant)

def correlation():
    score_by_model = retrieve_(name="score_by_model")
    # scores_by_user = retrieve_(name="scores_by_user")
    corr_by_model = {}
    corr_by_user = {}
    for c in score_by_model.keys():
        for m in score_by_model[c].keys():
            score_by_model[c][m] = np.array(score_by_model[c][m])
            score_by_model[c][m] = (score_by_model[c][m][::2] + score_by_model[c][m][1::2]) / 2
        score_by_model[c] = pd.DataFrame(score_by_model[c])
        corr_by_model[c] = score_by_model[c].corr()
        corr_by_user[c] = score_by_model[c].T.corr()
        plot_correlations(corr_by_model[c], f"model_correlation_{c}", f"{ctn(c)} scores correlated by model")
        plot_correlations(corr_by_user[c], f"user_correlation_{c}",
                          f"{ctn(c)} scores correlated by participant")
    dump_(name="score_dataframe", file=score_by_model)
    dump_(name="corr_by_model", file=corr_by_model)
    dump_(name="corr_by_user", file=corr_by_user)
    KMeans()

def plot_cross_correlations():
    plot_mean_cross_correlation()
    plot_mean_cross_correlation(False)

if __name__ == "__main__":
    plot_cross_correlations()
    # cross_criteria_correlation()
    # correlation()
    # test_re()
    # analyze_outputs(output_dir="./evaluation_output_backup")
    # analyze_scores(plot_model=False)
    # analyze_outputs("./output_backup_2504")
    # update_database(db="./output_backup_2504")
