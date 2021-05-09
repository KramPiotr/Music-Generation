import os
import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
import math
from utilities.notes_utils import notes_to_int_tuples
from utilities.utils import color_list, save_notes_and_durations, retrieve_notes_and_durations, print_to_file, \
    print_dict, retrieve_distincts_and_lookups, retrieve, save_fig, par_dir
from utilities.midi_utils import translate_chord
from utilities.part_utils import preprocess
from functools import partial
import matplotlib.patches as mpatches

plt.rcParams["font.family"] = "serif"

################################# Getters ###############################

def get_notes_durations(store_dir, kwargs=None):
    if kwargs is None:
        kwargs = {}
    if "notes" not in kwargs or "durations" not in kwargs:
        notes, durations = retrieve_notes_and_durations(store_dir)
    else:
        notes, durations = kwargs["notes"], kwargs["durations"]
    if len(notes) != len(durations):
        raise Exception(f"Number of notes {len(notes)} is not equal to the number of durations {len(durations)}")
    return notes, durations


def get_distincts_lookups(store_dir, kwargs):
    if "distincts" not in kwargs or "lookups" not in kwargs:
        distincts, lookups = retrieve_distincts_and_lookups(store_dir)
    else:
        distincts, lookups = kwargs["distincts"], kwargs["lookups"]
    return distincts, lookups


def get_notes(store_dir, kwargs):
    if "notes" not in kwargs:
        notes = retrieve(store_dir, "notes")
    else:
        notes = kwargs["notes"]
    return notes


def get_durations(store_dir, kwargs):
    if "durations" not in kwargs:
        durations = retrieve(store_dir, "durations")
    else:
        durations = kwargs["durations"]
    return durations


################################# Basic analysis ###############################

def basic_analysis_no_dl(store_dir, analyze_dir):  # dl = distincts and lookups
    notes, durations = retrieve_notes_and_durations(store_dir)
    notes = notes_to_int_tuples(notes)
    if len(notes) != len(durations):
        raise Exception(f"Number of notes {len(notes)} is not equal to the number of durations {len(durations)}")
    distinct_notes = len(set(notes))
    distinct_durations = len(set(durations))
    os.makedirs(analyze_dir, exist_ok=True)
    with open(os.path.join(analyze_dir, "basic_analysis.txt"), "w") as f:
        f.write(f"Number of pairs of notes and durations {len(notes)}")
        f.write(f"\nNumber of distinct notes {distinct_notes}")
        f.write(f"\nNumber of distinct durations {distinct_durations}")


def describe_dataset(analyze_dir, **kwargs):
    notes, durations = get_notes_durations(par_dir(analyze_dir), kwargs)
    distincts, lookups = get_distincts_lookups(par_dir(analyze_dir), kwargs)
    with open(os.path.join(analyze_dir, "description.txt"), "w") as f:
        print = partial(print_to_file, file=f)
        print(f"Description of dataset: {par_dir(analyze_dir)}")
        print(f"Number of pairs of notes and durations {len(notes)}")
        if len(distincts) == 4:
            (notes_names, n_notes, duration_names, n_durations), (
            note_to_int, int_to_note, duration_to_int, int_to_duration) = distincts, lookups
            print(f"Number of distinct notes: {n_notes}")
            print(f"Number of distinct durations: {n_durations}\n")
            print("Notes")
            print_dict(int_to_note, print)
            print("\nDurations")
            print_dict(int_to_duration, print)
        else:
            (duration_name, n_durations), (duration_to_int, int_to_duration) = distincts, lookups
            print(f"Number of durations: {n_durations}\n")
            print("Durations")
            print_dict(int_to_duration, print)


################################# Pitches analysis ###############################

def plot_frequencies_by_pitch(analyze_dir, suffix="", **kwargs):
    notes = get_notes(par_dir(analyze_dir), kwargs)
    n_per_pitch = {}
    for note in notes:
        for pitch in note:
            n_per_pitch[pitch] = n_per_pitch.get(pitch, 0) + 1

    pitches = sorted(n_per_pitch.keys())
    occurence = [n_per_pitch[p] for p in pitches]

    xticks = [-1, 0, 2, 4, 5, 7, 9, 11]
    xlabels = ["S", "C", "D", "E", "F", "G", "A", "B"]

    plt.figure()

    def make_plot():
        plt.bar(pitches, occurence, color=color_list(pitches, 2), width=1)
        plt.title("Pitch frequency")
        plt.xlabel("Pitch")
        plt.ylabel("Number of occurences")
        plt.xticks(xticks, xlabels)

    make_plot()
    save_fig(plt, analyze_dir, "pitch_frequency" + suffix)
    return make_plot

def zipfs_plots(chords, n_per_note, analyze_dir):
    plt.figure(figsize=(10, 10))
    chords = sorted(n_per_note.keys(), key=lambda c: -n_per_note[c])
    cs = [c for c in chords if len(c) > 1]
    ys = [n_per_note[c] for c in cs]
    plt.scatter(np.arange(1, len(cs) + 1), ys, color='blue')
    plt.title(f"Chord frequency")
    plt.xlabel("Log of the rank number")
    plt.ylabel("Log of the number of occurences")
    plt.yscale("log")
    plt.xscale("log")
    # for i in range(1, 5):
    #     plt.subplot(2, 2, i)
    #     cs = [c for c in chords if len(c)==i][::-1]
    #     ys = [n_per_note[c] for c in cs]
    #     plt.scatter(np.arange(1, len(cs) + 1), ys, color='blue')
    #     plt.title(f"Chords with {i} pitches")
    #     plt.xlabel("Log of the rank number")
    #     plt.ylabel("Log of the number of occurences")
    #     plt.yscale("log")
    #     plt.xscale("log")
    save_fig(plt, analyze_dir, "zipf_plots")
    plt.show()



def analyze_frequencies_by_group(analyze_dir, suffix="", n_max_chords=10, n_labels=None, **kwargs):
    np.random.seed(0)
    notes = get_notes(par_dir(analyze_dir), kwargs)
    notes = notes_to_int_tuples(notes)
    chords = notes
    n_per_note = {}
    n_per_len = {}
    for note in chords:
        if note[0] != -1:
            n_per_note[note] = n_per_note.get(note, 0) + 1
            n_per_len[len(note)] = n_per_len.get(len(note), 0) + 1

    chords = sorted(n_per_note.keys(), key=lambda c: (len(c), n_per_note[c]))
    zipfs_plots(chords, n_per_note, analyze_dir)
    occurence = [n_per_note[c] for c in chords]
    colors = color_list([len(c) for c in chords], 2)

    maxes = [-1]
    max_chords = []
    for i in range(len(chords)):
        if i == len(chords) - 1 or (len(chords[i]) < len(chords[i + 1])):
            maxes.append(i)
            max_chords.append((len(chords[i]),
                               [(translate_chord(c), n_per_note[c]) for c in chords[i:max(0, i - n_max_chords):-1] if
                                len(c) == len(chords[i])]))

    with open(os.path.join(analyze_dir, "frequent_chords" + suffix + ".txt"), "w") as f:
        for length, best_chords in max_chords:
            f.write(f"\n----------------------------{length}----------------------------\n")
            for i, (chord, occur) in enumerate(best_chords):
                f.write(
                    ('%-40s' % f"    {i + 1}. {chord} ") + f"{occur}\n")  # you can change up to something easy to read

    xticks = []
    lengths = sorted(n_per_len.keys())
    xlabels = lengths[:n_labels] if n_labels is not None else lengths
    for i in range(1, len(maxes)):
        xticks.append(math.ceil((maxes[i - 1] + maxes[i]) / 2))

    plt.figure()

    def make_plot1(colors):
        plt.bar(np.arange(len(chords)), occurence, color=colors, width=1)
        plt.title("Sorted frequencies of chords grouped by length")
        plt.xlabel("Length of the chords in each group")
        plt.ylabel("Number of occurences")
        plt.yscale("log")
        plt.xticks(xticks, xlabels)

    make_plot1 = partial(make_plot1, colors)
    make_plot1()
    save_fig(plt, analyze_dir, "sorted_frequencies_by_group" + suffix)

    y = [n_per_len[l] for l in lengths]
    x = np.arange(len(y))
    colors = color_list(x, 2)
    plt.figure()

    def make_plot2():
        plt.bar(x, y, color=colors, width=1)
        plt.title("Frequency of a chord group")
        plt.xlabel("Length of the chords in each group")
        plt.ylabel("Number of occurences")
        plt.yscale("log")
        plt.xticks(x, lengths)

    make_plot2()
    save_fig(plt, analyze_dir, "frequencies_by_group" + suffix)

    return make_plot1, make_plot2


def plot_frequent_chords():
    def process(line):
        line = line.replace('-', 'b')
        y = int(line[30:])
        part1 = line[:30].split(".")[1]
        splitted = part1.split("bb")
        if len(splitted) == 1:
            return (splitted[0].strip(), y)
        else:
            return (splitted[0].strip(), y, splitted[1].strip())

    analysis_dir = '..\\run\\two_datasets_store\\version_3\\analysis'
    fig = plt.figure(figsize=(10, 6))
    plt.suptitle("Most frequent chords grouped by length", fontsize=17)
    patches = [mpatches.Patch(color='black', label='font color for pitches'),
               mpatches.Patch(color='#D0171F', label='font color for music chords')]
    fig.legend(handles=patches)
    for i in range(1, 5):
        fc = os.path.join(analysis_dir, f'fc{i}.txt')
        with open(fc, 'r') as f:
            lines = f.readlines()
            data = [process(line) for line in lines]
            labels = [x[0] for x in data]
            ys = [x[1] for x in data]
            plt.subplot(2, 2, i)
            color = ['#FF993380', '#FF660080']
            if i < 3:
                color = [c[:-2] for c in color]
            bars = plt.bar(np.arange(len(ys)), ys, tick_label=labels, color=color)
            if i > 2:
                bar_labels = [x[2] for x in data]
                plt.bar_label(bars, labels=bar_labels, label_type='center', color='#D0171F', fontweight='bold', fontsize=8)
            if i > 1:
                plt.xticks(rotation=30)
    save_fig(plt, analysis_dir, "frequency_length")
    plt.show()
    #for fc in ['fc1.txt', 'fc2.txt', 'fc3.txt', 'fc4.txt']


################################# Durations analysis ###############################

# TODO remember incorporate notes changes into durations changes

def analyze_frequencies_by_duration(analyze_dir, suffix="", space_labels=2, **kwargs):
    durations = get_durations(par_dir(analyze_dir), kwargs)
    n_per_duration = {}
    seq_of_zeros = []
    for i, d in enumerate(durations):
        n_per_duration[d] = n_per_duration.get(d, 0) + 1
        if d == 0:
            if len(seq_of_zeros) > 0 and seq_of_zeros[-1][1] == i - 1:
                seq_of_zeros[-1][1] += 1
            else:
                seq_of_zeros.append([i, i])

    # for seq in seq_of_zeros:
    #     if seq[1] - seq[0] != 31: #32 is the sequence length used for splitting the songs
    #         print(f"invalid 0s sequence: {seq}")

    durations = sorted(n_per_duration.keys())
    occurence = [n_per_duration[d] for d in durations]
    x = np.arange(len(durations))

    sorted_occurences = sorted(n_per_duration.items(), key=lambda x: x[1], reverse=True)
    with open(os.path.join(analyze_dir, "durations_by_frequency" + suffix + ".txt"), "w") as f:
        for i, (dur, occur) in enumerate(sorted_occurences):
            f.write(('%-40s' % f"    {i}. {dur} ") + f"{occur}\n")  # you can change up to something easy to read

    rc('text', usetex=True)

    plt.figure()

    xticks = [0, 1, 3, 6, 9, 12, 15, 18]

    def frac(a, b):
        return r"$\frac{"+str(a)+"}{"+str(b)+"}$"

    #xlabels = ["S", "1/4", "1/2", "1", "3/2", "2", "5/2", "3"]
    xlabels = ["S", frac(1, 4), frac(1, 2), "1", frac(3, 2), "2", frac(5, 2), "3"]

    #xlabels = [f"${x}$" for x in xlabels]

    def make_plot():
        plt.bar(x, occurence, color=color_list(x, 2), width=1)
        plt.title("Duration frequency")
        plt.xlabel("Duration (in quarter notes)")
        plt.ylabel("Number of occurences")
        plt.yscale("log")
        plt.xticks(xticks, xlabels)
        #plt.xticks(x[::space_labels], durations[::space_labels])

    make_plot()
    save_fig(plt, analyze_dir, "duration_frequency" + suffix)
    return make_plot


def create_processed(store_folder, notes, durations):
    processed_store = os.path.join(store_folder, "processed")
    os.makedirs(processed_store)
    notes, durations = preprocess(notes, durations)
    save_notes_and_durations(processed_store, notes, durations)


def analyze_dataset(dataset_dir):
    analyze_dir = os.path.join(dataset_dir, "analysis")
    os.makedirs(analyze_dir, exist_ok=True)
    notes, durations = retrieve_notes_and_durations(dataset_dir)
    distincts, lookups = retrieve_distincts_and_lookups(dataset_dir)
    unpacked_data = {
        "notes": notes,
        "durations": durations,
        "distincts": distincts,
        "lookups": lookups,
    }
    describe_dataset(analyze_dir, **unpacked_data)
    if "attention" not in dataset_dir:
        freq_by_pitch_plot = plot_frequencies_by_pitch(analyze_dir, **unpacked_data)
        zipf_plot, aggregate_freq_plot = analyze_frequencies_by_group(analyze_dir, **unpacked_data)
        freq_by_dur_plot = analyze_frequencies_by_duration(analyze_dir, **unpacked_data)
        plt.figure(figsize=(10, 7.5))
        plt.subplot(2, 2, 1)
        freq_by_pitch_plot()
        plt.subplot(2, 2, 2)
        freq_by_dur_plot()
        plt.subplot(2, 2, 3)
        aggregate_freq_plot()
        plt.subplot(2, 2, 4)
        zipf_plot()
        plt.tight_layout(2)
        save_fig(plt, analyze_dir, "dataset_visualization")


def check_for_rests(dataset_dir):
    notes, durations = retrieve_notes_and_durations(dataset_dir)
    rs = []
    s = []
    for i in range(len(notes)):
        if notes[i][0] == -1:
            if durations[i] == 0:
                s.append(i)
            rs.append(i)

    assert (len(s) == len(rs))  # There is no rests in the dataset

def analyze_dataset_by_version(version=3):
    analyze_dataset(f"../run/two_datasets_store/version_{version}")

def change_font():
    plt.rcParams["font.family"] = "serif"
    analyze_dataset_by_version()
    plot_frequent_chords()

if __name__ == "__main__":
    analyze_frequencies_by_group(f"../run/two_datasets_store/version_3/analysis")
    # change_font()
    # plot_frequent_chords()
    # analyze_dataset_by_version()
    # pass
    # check_for_rests("../run/two_datasets_store/version_1")
    # analyze_dataset("../run/two_datasets_store/version_0")
