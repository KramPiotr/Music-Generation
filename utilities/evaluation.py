from glob import glob
import os
from tqdm import tqdm
from utilities.utils import dump, retrieve, save_fig
from statistics import mean, stdev
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import re

evaluation_dir = "./evaluation_output"
pickle_dir = os.path.join(evaluation_dir, "pickles")
os.makedirs(pickle_dir, exist_ok=True)


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


def analyze_outputs(output_dir="./example_evaluation_output"):
    global evaluation_dir
    output_files = list(glob(os.path.join(output_dir, "output*")))
    scores = []
    comments = []
    for file in tqdm(output_files):
        with open(file, "r") as f:
            output_lines = [line.split(";") for line in f.read().split("\n")]
            comment = ""
            for line in output_lines:
                if len(line) == 4 and line[0].startswith("assets/"):
                    scores.append(extract_score(line))
                else:
                    if comment != "":
                        comment += "\n"
                    comment += ";".join(line)
            if comment != "":
                comments.append(comment)
    all_comments = ("\n" + "-" * 50 + "\n").join(comments)
    os.makedirs(evaluation_dir, exist_ok=True)

    with open(os.path.join(evaluation_dir, "comments.txt"), "w") as f:
        f.write(all_comments)

    save_scores(scores)


def save_scores(scores):
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
    if model=="MusicVAE":
        return name[-8:-4]
    if model=="Bach" or model=="Random":
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
        plot_means_by_item(mean_by_model, std_by_model, name="mean_scores_by_model", title="Mean scores and their standard deviations by model")

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
                                       legend=i==0,
                                       save=False,
                                       show=False,
                                       model=m,
                                       figsize=(12, 6))
        make_plot()
    plt.tight_layout()
    save_fig(plt, evaluation_dir, "mean_scores_by_song")
    plt.show()

def test_re():
    print(re.search("-\d\d_\d\d_\d\d", "output-2021_04_17--02_44_22-None-4-1-weights-improvement-09-5.7726").group()[1:])
    print(extract_label("sdfsdf.mp3", "Bach"))

if __name__ == "__main__":
    # test_re()
    # analyze_outputs(output_dir="./evaluation_output_backup")
    analyze_scores(plot_model=False)
